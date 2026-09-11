import os        # 文件系统操作 (主要用 os.fsync 刷盘)
import pathlib    # 路径处理, 用 Path 对象
import modal      # Modal SDK, 上云核心

app = modal.App("alphafold3-batch")  # 创建 App, 所有函数 / 入口都挂在上面

# ============================================================
# 本地路径配置
# ============================================================
INPUT_DIR = pathlib.Path(r"G:\Lab_Data\af3_inputs")                    # 输入: 原始序列 JSON
MSA_DIR = pathlib.Path(r"G:\Lab_Data\af3_msa")                         # 缓存: data pipeline 产物 _data.json
MSA_OUTPUT_DIR = pathlib.Path(r"G:\Lab_Data\af3_msa_outputs")          # 输出: MSA 推理结果
NO_MSA_DIR = pathlib.Path(r"G:\Lab_Data\af3_no_msa")                   # 中间: MSA-free 输入 (补空 MSA 字段)
NO_MSA_OUTPUT_DIR = pathlib.Path(r"G:\Lab_Data\af3_no_msa_outputs")    # 输出: MSA-free 推理结果


# ============================================================
# 严格对照 AlphaFold3 官方 Dockerfile 构建 af3_image
# ============================================================
af3_image = (
    modal.Image.from_registry(                       # 基础镜像: 官方 CUDA 12.6.3 + Ubuntu 24.04
        "nvidia/cuda:12.6.3-base-ubuntu24.04",
        add_python="3.12",                           # 镜像内装 Python 3.12
    )
    .apt_install(                                    # 系统依赖: git/wget + 编译工具链 + zlib/zstd/patch
        "git", "wget",
        "gcc", "g++", "make",
        "zlib1g-dev", "zstd",
        "patch", "clang",
    )
    .pip_install("uv==0.9.24")                       # 装 uv (AF3 官方用的包管理器), 锁版本
    .env({                                           # uv 行为 + PATH (hmmer 和 venv 放最前)
        "UV_COMPILE_BYTECODE": "1",
        "UV_PROJECT_ENVIRONMENT": "/alphafold3_venv",
        "PATH": "/hmmer/bin:/alphafold3_venv/bin:/usr/local/bin:/usr/bin:/bin",
    })
    .run_commands("uv venv /alphafold3_venv")        # 建独立虚拟环境
    .run_commands(
        "git clone https://github.com/google-deepmind/alphafold3.git /app/alphafold",  # 拉 AF3 源码
    )
    .run_commands(                                   # 下载 hmmer 3.4 源码 + 校验 sha256 + 解压
        "mkdir -p /hmmer_build /hmmer",
        "wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz -P /hmmer_build",
        "cd /hmmer_build && echo 'ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3  hmmer-3.4.tar.gz' | sha256sum --check",
        "cd /hmmer_build && tar zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz",
    )
    .run_commands(                                   # 打 AF3 官方补丁 (jackhmmer 序列数上限)
        "cp /app/alphafold/docker/jackhmmer_seq_limit.patch /hmmer_build/",
        "cd /hmmer_build && patch -p0 < jackhmmer_seq_limit.patch",
    )
    .run_commands(                                   # 编译安装 hmmer 到 /hmmer, 装完删源码
        "cd /hmmer_build/hmmer-3.4 && ./configure --prefix=/hmmer && make -j4",
        "cd /hmmer_build/hmmer-3.4 && make install",
        "cd /hmmer_build/hmmer-3.4/easel && make install",
        "rm -rf /hmmer_build",
    )
    .run_commands(                                   # 按 uv.lock 装 AF3 全部依赖 (--frozen 不改锁文件)
        "cd /app/alphafold && UV_HTTP_TIMEOUT=1800 uv sync --frozen --all-groups --no-editable",
    )
    .run_commands(                                   # 编译 AF3 的数据处理扩展 (build_data)
        "cd /app/alphafold && uv run build_data",
    )
    .env({                                           # XLA/JAX 运行期调优 (关 triton gemm + 预占 95% 显存)
        "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_CLIENT_MEM_FRACTION": "0.95",
    })
)


# ============================================================
# Volumes: 数据库/权重 + MSA 缓存 + 推理结果
# ============================================================
af3_volume = modal.Volume.from_name("alphafold3-data")  # 数据库 + 权重 (需预先建好, 这里不自动创建)

msa_cache_volume = modal.Volume.from_name(   # MSA 缓存 (data pipeline 产物)
    "alphafold3-msa-cache",
    create_if_missing=True,                  # 不存在就自动创建
)

results_volume = modal.Volume.from_name(     # 推理结果
    "alphafold3-results",
    create_if_missing=True,
)


# ============================================================
# MSA 缓存路径约定 (AF3 原生结构, 本地和 volume 一致)
#   子文件夹: {job_name}/
#   缓存文件: {job_name}/{job_name}_data.json
# ============================================================
def msa_file_name(job_name: str) -> str:     # job 名 -> data.json 文件名
    return f"{job_name}_data.json"


def msa_remote_path(job_name: str) -> str:   # job 名 -> volume 内相对路径 {job}/{job}_data.json
    return f"{job_name}/{msa_file_name(job_name)}"


# ============================================================
# 函数 1: 数据管线阶段 (MSA + 模板搜索)
# ============================================================
@app.function(
    image=af3_image,                         # 用上面构建的镜像
    volumes={
        "/data": af3_volume,                 # 数据库/权重 (只读)
        "/msa_cache": msa_cache_volume,      # MSA 缓存写出处
    },
    cpu=16,                                  # 16 核 CPU
    memory=8192,                            # 8 GB 内存
    timeout=60 * 60 * 12,                    # 最长 12 小时
)
def run_data_pipeline(fasta_json: str, job_name: str) -> str:
    import subprocess                        # 容器内 import (云端使用)
    import pathlib
    import shutil
    import os

    # 缓存目标: /msa_cache/{job}/{job}_data.json (AF3 原生结构)
    target_dir = pathlib.Path(f"/msa_cache/{job_name}")
    target_file = target_dir / msa_file_name(job_name)

    msa_cache_volume.reload()                # 拉取 volume 最新状态
    if target_file.exists():                 # 已有缓存 -> 直接跳过
        print(f"[cache hit] job={job_name}")
        return job_name

    print(f"[cache miss] job={job_name}, running data pipeline...")

    # 输入 JSON 写到容器内临时路径
    input_dir = pathlib.Path("/tmp/af_input")
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / f"{job_name}.json"
    input_path.write_text(fasta_json)

    # AF3 先写容器内临时目录, 跑完再搬到 volume
    tmp_out = pathlib.Path(f"/tmp/af_out/{job_name}")
    if tmp_out.exists():
        shutil.rmtree(tmp_out)
    tmp_out.mkdir(parents=True, exist_ok=True)

    cmd = [                                  # 调 AF3 主程序
        "/alphafold3_venv/bin/python3", "/app/alphafold/run_alphafold.py",
        f"--json_path={input_path}",
        "--db_dir=/data/databases",          # 数据库目录
        f"--output_dir={tmp_out}",
        "--norun_inference",                 # 关键: 只跑数据管线, 不做推理
        "--jackhmmer_n_cpu=6",               # jackhmmer 用 6 核
    ]
    subprocess.run(cmd, check=True, cwd="/app/alphafold")  # check=True: 非零退出即抛错

    # 找 AF3 产出的 *_data.json (含 MSA + 模板)
    data_jsons = list(tmp_out.rglob("*_data.json"))
    if not data_jsons:
        raise FileNotFoundError(f"No *_data.json produced by AF3 in {tmp_out}")
    source = data_jsons[0]

    target_dir.mkdir(parents=True, exist_ok=True)
    # 分块写 + fsync 强制刷盘, 避免在 Modal volume (FUSE) 上写出损坏文件
    with open(source, "rb") as src_f, open(target_file, "wb") as dst_f:
        shutil.copyfileobj(src_f, dst_f, length=1024 * 1024)
        dst_f.flush()
        os.fsync(dst_f.fileno())

    # 校验大小, 防止只写了一半
    src_size = source.stat().st_size
    dst_size = target_file.stat().st_size
    if src_size != dst_size:
        raise RuntimeError(
            f"MSA cache write size mismatch for {job_name}: "
            f"src={src_size} dst={dst_size}"
        )

    shutil.rmtree(tmp_out, ignore_errors=True)  # 清临时目录
    msa_cache_volume.commit()                # 提交 volume, 持久化缓存
    print(f"[done] MSA cached at {target_file}")
    return job_name


# ============================================================
# 函数 2: 推理阶段
# 读取: /msa_cache/{job_name}/{job_name}_data.json
# 产出: /results/{job_name}/...
# ============================================================
@app.function(
    image=af3_image,
    volumes={
        "/data": af3_volume,
        "/msa_cache": msa_cache_volume,
        "/results": results_volume,
    },
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=60 * 60 * 12,
    retries=modal.Retries(max_retries=2, initial_delay=10.0),   # 偶发网络故障自愈, 不拖垮整批
)
def run_inference(job_name: str) -> str:
    import subprocess
    import pathlib
    import shutil

    msa_cache_volume.reload()

    data_json_path = pathlib.Path(f"/msa_cache/{msa_remote_path(job_name)}")

    if not data_json_path.exists():
        raise FileNotFoundError(
            f"MSA cache not found at {data_json_path}. "
            f"Did you run run_data_pipeline first?"
        )

    print(f"[{job_name}] Using MSA data file: {data_json_path}")

    result_dir = pathlib.Path(f"/results/{job_name}")
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # 直接用镜像里装好的 venv, 不走 uv run —— uv run 每次会校验并可能重建环境,
    # 重建时 cifpp 的 CMake 要联网下 CCD 文件, SSL 一挂整批任务就中止
    cmd = [
        "/alphafold3_venv/bin/python3", "/app/alphafold/run_alphafold.py",
        f"--json_path={data_json_path}",
        "--model_dir=/data/parameters",
        f"--output_dir={result_dir}",
        "--norun_data_pipeline",
    ]
    subprocess.run(cmd, check=True, cwd="/app/alphafold")

    results_volume.commit()
    print(f"[{job_name}] Inference done, results at /results/{job_name}")
    return str(result_dir)


# ============================================================
# 函数 3: MSA-free 推理 (跳过 data pipeline, 直接用空 MSA)
# 读取: 原始序列 JSON 字符串 (容器内补齐空 MSA/templates 字段)
# 产出: /results/{job_name}/...
# 精度会明显下降,适用于快速筛查/孤儿蛋白/de novo 设计蛋白
# ============================================================
@app.function(
    image=af3_image,
    volumes={
        "/data": af3_volume,
        "/results": results_volume,
    },
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=60 * 60 * 12,
)
def run_inference_no_msa(job_name: str, raw_json: str) -> str:
    import json
    import subprocess
    import pathlib
    import shutil

    # 解析原始 JSON, 给每个 protein 条目补齐 MSA-free 必需字段
    data = json.loads(raw_json)
    for entry in data.get("sequences", []):
        if "protein" in entry:
            protein = entry["protein"]
            protein.setdefault("modifications", [])
            protein.setdefault("unpairedMsa", "")
            protein.setdefault("pairedMsa", "")
            protein.setdefault("templates", [])

    tmp_dir = pathlib.Path("/tmp/af_nomsa_input")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_json = tmp_dir / f"{job_name}.json"
    tmp_json.write_text(json.dumps(data))

    result_dir = pathlib.Path(f"/results/{job_name}")
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "/alphafold3_venv/bin/python3", "/app/alphafold/run_alphafold.py",
        f"--json_path={tmp_json}",
        "--model_dir=/data/parameters",
        f"--output_dir={result_dir}",
        "--norun_data_pipeline",
    ]
    subprocess.run(cmd, check=True, cwd="/app/alphafold")

    results_volume.commit()
    print(f"[{job_name}] MSA-free inference done, results at /results/{job_name}")
    return str(result_dir)


# ============================================================
# 本地辅助: volume ↔ 本地 文件传输
# ============================================================
def is_bulky_result(file_name: str) -> bool:
    """推理结果里体积大但通常用不到的文件, 需要时再手动去 volume 取。

    *_confidences.json      每 sample 约 6.4 MB, 完整 PAE 矩阵 (注意别误伤 *_summary_confidences.json)
    *_data.json             约 40 MB, AF3 把输入 MSA 原样复制到输出, 本地 MSA_DIR 已有同一份
    """
    if file_name.endswith("_data.json"):
        return True
    return (file_name.endswith("_confidences.json")
            and not file_name.endswith("_summary_confidences.json"))


def download_from_volume(volume: modal.Volume, remote_prefix: str, local_dir: pathlib.Path,
                         skip_bulky: bool = False, entries=None) -> int:
    """把 volume 下 remote_prefix 目录递归下载到 local_dir

    - 每个文件先写到 .part, 完整且大小匹配后再 rename 到正式名
    - 单文件失败不影响其他文件, 打日志继续
    - 写入后 fsync, 避免 OS 级缓存未刷盘
    - skip_bulky: 跳过 is_bulky_result() 命中的大文件 (只对推理结果开, MSA 缓存不能开)
    - 返回成功下载的文件数
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    prefix = remote_prefix.rstrip("/")
    if entries is None:                  # 调用方没预先列好就自己列 (保持原行为)
        try:
            entries = list(volume.iterdir(f"{prefix}/", recursive=True))
        except (FileNotFoundError, modal.exception.NotFoundError):
            return 0

    success = 0
    skipped = 0
    failed = []
    for entry in entries:
        if entry.type != modal.volume.FileEntryType.FILE:
            continue
        if skip_bulky and is_bulky_result(pathlib.Path(entry.path).name):
            skipped += 1
            continue
        rel_path = pathlib.Path(entry.path).relative_to(prefix)
        local_path = local_dir / rel_path
        tmp_path = local_path.with_name(local_path.name + ".part")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        expected_size = getattr(entry, "size", None)

        try:
            with open(tmp_path, "wb") as f:
                for chunk in volume.read_file(entry.path):
                    f.write(chunk)
                f.flush()
                os.fsync(f.fileno())

            actual_size = tmp_path.stat().st_size
            if expected_size is not None and actual_size != expected_size:
                raise IOError(
                    f"size mismatch: expected={expected_size} got={actual_size}"
                )

            tmp_path.replace(local_path)
            success += 1
            size_info = f"{actual_size} bytes"
            print(f"    [OK]   {entry.path} ({size_info})")

        except Exception as e:
            failed.append((entry.path, repr(e)))
            print(f"    [FAIL] {entry.path}: {e}")
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass

    if skipped:
        print(f"    [skip] {skipped} 个大文件未下载 (需要时去 volume 取)")
    if failed:
        print(f"  [WARN] {len(failed)} file(s) failed under prefix '{prefix}'")
    return success


def upload_file_to_volume(volume: modal.Volume, local_file: pathlib.Path, remote_path: str) -> None:
    """把单个 local_file 上传到 volume 的 remote_path (覆盖)"""
    with volume.batch_upload(force=True) as batch:
        batch.put_file(str(local_file), remote_path)


def transform_to_msa_free(raw_json: str) -> str:
    """给每个 protein 条目补齐空 MSA/templates/modifications 字段"""
    import json
    data = json.loads(raw_json)
    for entry in data.get("sequences", []):
        if "protein" in entry:
            protein = entry["protein"]
            protein.setdefault("modifications", [])
            protein.setdefault("unpairedMsa", "")
            protein.setdefault("pairedMsa", "")
            protein.setdefault("templates", [])
    return json.dumps(data, indent=2, ensure_ascii=False)


def volume_has_msa_cache(job_name: str) -> bool:
    """检查 volume 里是否已有该 job 的 MSA 缓存文件 (本地调用)"""
    target_path = msa_remote_path(job_name)  # {job}/{job}_data.json
    try:
        for entry in msa_cache_volume.iterdir(f"{job_name}/", recursive=True):
            if entry.type == modal.volume.FileEntryType.FILE and entry.path == target_path:
                return True
    except (FileNotFoundError, modal.exception.NotFoundError):
        return False
    return False


# ============================================================
# 入口 1: 完整流水线 (data pipeline + inference + 下载本地)
# 用法: modal run af3_modal.py::main
# ============================================================
@app.local_entrypoint()
def main(skip_existing: bool = True):
    """
    完整批量流水线:
      1. 扫描 INPUT_DIR 下所有 .json
      2. volume 里有缓存的跳过 data pipeline,没有的跑 data pipeline
      3. 下载 MSA 缓存到本地 MSA_DIR
      4. 对所有 job 跑 inference
      5. 下载推理结果到本地 MSA_OUTPUT_DIR

    skip_existing: 本地已存在结果目录的 job 跳过 (默认 True)
    """
    import concurrent.futures

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}\n"
            f"请在脚本顶部修改 INPUT_DIR,或创建这个文件夹"
        )

    json_files = sorted(INPUT_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {INPUT_DIR}")

    jobs = []
    # 以存在非空 {job}_model.cif 为完成标志, 0 字节或缺失都视为未完成
    for jf in json_files:
        job_name = jf.stem
        job_dir = MSA_OUTPUT_DIR / job_name
        marker_files = list(job_dir.rglob(f"{job_name}_model.cif")) if job_dir.exists() else []
        if skip_existing and any(m.stat().st_size > 0 for m in marker_files):
            print(f"[skip] {job_name} already has complete local results")
            continue
        jobs.append((job_name, jf.read_text(encoding="utf-8")))

    if not jobs:
        print("Nothing to do.")
        return

    print("=" * 60)
    print(f"Found {len(json_files)} input(s), {len(jobs)} to process")
    print(f"Input dir:  {INPUT_DIR}")
    print(f"MSA cache:  {MSA_DIR}")
    print(f"Output dir: {MSA_OUTPUT_DIR}")
    print("=" * 60)

    # --- 检查 volume MSA 缓存 ---
    print("\n[Stage 1/3] Checking volume MSA cache...")
    cached = []
    uncached = []
    for job_name, fasta_json in jobs:
        if volume_has_msa_cache(job_name):
            cached.append((job_name, fasta_json))
            print(f"  [HIT]  {job_name}")
        else:
            uncached.append((job_name, fasta_json))
            print(f"  [MISS] {job_name}")

    # --- 阶段 1: data pipeline (只跑未命中的) ---
    if uncached:
        print(f"\n[Stage 2/3] Running data pipeline for {len(uncached)} job(s)")
        print("=" * 60)
        args = [(fj, jn) for jn, fj in uncached]
        list(run_data_pipeline.starmap(args, order_outputs=True))
    else:
        print(f"\n[Stage 2/3] All {len(jobs)} job(s) cached, skip data pipeline")

    # --- 下载 MSA 缓存到本地 (所有 job) ---
    print(f"\n[Download MSA] -> {MSA_DIR}")
    MSA_DIR.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(
                download_from_volume,
                msa_cache_volume,
                job_name,
                MSA_DIR / job_name,
            ): job_name
            for job_name, _ in jobs
        }
        for fut in concurrent.futures.as_completed(futures):
            job_name = futures[fut]
            try:
                n = fut.result()
                print(f"  [OK]   {job_name:20s} {n} MSA file(s)")
            except Exception as e:
                print(f"  [FAIL] {job_name:20s} download failed: {e}")

    # --- 阶段 2: inference ---
    print(f"\n[Stage 3/3] Running inference for {len(jobs)} job(s)")
    print("=" * 60)
    inf_args = [(job_name,) for job_name, _ in jobs]
    list(run_inference.starmap(inf_args, order_outputs=True))

    # --- 下载推理结果到本地 ---
    print(f"\n[Download Results] -> {MSA_OUTPUT_DIR}")
    MSA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(
                download_from_volume,
                results_volume,
                job_name,
                MSA_OUTPUT_DIR / job_name,
                True,          # skip_bulky: 不下 *_confidences.json 与 *_data.json
            ): job_name
            for job_name, _ in jobs
        }
        for fut in concurrent.futures.as_completed(futures):
            job_name = futures[fut]
            try:
                n = fut.result()
                print(f"  [OK]   {job_name:20s} {n} files")
            except Exception as e:
                print(f"  [FAIL] {job_name:20s} download failed: {e}")

    print("\n" + "=" * 60)
    print("All done.")
    print(f"  MSA cache: {MSA_DIR}")
    print(f"  Results:   {MSA_OUTPUT_DIR}")
    print("=" * 60)


# ============================================================
# 入口 2: 只跑 data pipeline (本地 + volume 都存一份)
# 用法: modal run af3_modal.py::only_data_pipeline
# ============================================================
@app.local_entrypoint()
def only_data_pipeline(skip_existing: bool = True):
    """
    扫描 INPUT_DIR 下所有 .json,只跑 data pipeline:
      1. volume 里没缓存的跑 data pipeline
      2. 所有 job 的缓存都同步一份到本地 MSA_DIR

    skip_existing: volume 已有缓存的 job 跳过 (默认 True)
    """
    import concurrent.futures

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    json_files = sorted(INPUT_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {INPUT_DIR}")

    all_job_names = [jf.stem for jf in json_files]

    jobs = []
    for jf in json_files:
        job_name = jf.stem
        if skip_existing and volume_has_msa_cache(job_name):
            print(f"[skip] {job_name} already cached in volume")
            continue
        jobs.append((job_name, jf.read_text(encoding="utf-8")))

    print("=" * 60)
    print(f"Found {len(json_files)} input(s), {len(jobs)} to run data pipeline")
    print("=" * 60)

    if jobs:
        args = [(fj, jn) for jn, fj in jobs]
        list(run_data_pipeline.starmap(args, order_outputs=True))
    else:
        print("All inputs already cached, no data pipeline to run.")

    # 下载所有 job 的缓存到本地
    print(f"\n[Download MSA] -> {MSA_DIR}")
    MSA_DIR.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(
                download_from_volume,
                msa_cache_volume,
                job_name,
                MSA_DIR / job_name,
            ): job_name
            for job_name in all_job_names
        }
        for fut in concurrent.futures.as_completed(futures):
            job_name = futures[fut]
            try:
                n = fut.result()
                print(f"  [OK]   {job_name:20s} {n} MSA file(s)")
            except Exception as e:
                print(f"  [FAIL] {job_name:20s} download failed: {e}")

    print("Data pipeline done.")


# ============================================================
# 入口 3: 只跑 inference (从本地 MSA 缓存上传,再推理)
# 用法: modal run af3_modal.py::only_inference
# ============================================================
@app.local_entrypoint()
def only_inference(skip_existing: bool = True):
    """
    扫描本地 MSA_DIR 下所有 {job}/{job}_data.json (AF3 原生结构):
      1. 把本地 data.json 上传到 volume (volume 已有则跳过上传)
      2. 跑 inference
      3. 下载推理结果到本地 MSA_OUTPUT_DIR

    skip_existing: 本地已存在结果目录的 job 跳过 (默认 True)
    """
    import concurrent.futures

    if not MSA_DIR.exists():
        raise FileNotFoundError(f"Local MSA cache dir not found: {MSA_DIR}")

    # 扫描 AF3 原生结构: 每个子文件夹里找 *_data.json, job 名 = 文件夹名
    found = []
    for d in sorted(MSA_DIR.iterdir()):
        if not d.is_dir():
            continue
        data_files = sorted(d.glob("*_data.json"))
        if not data_files:
            print(f"[skip] {d.name} 文件夹内无 *_data.json")
            continue
        found.append((d.name, data_files[0]))    # (job 名, data.json 路径)

    if not found:
        print(f"No valid MSA folders in {MSA_DIR}")
        return

    # 过滤已有本地结果的 (以存在非空 {job}_model.cif 为完成标志, 0 字节或缺失都视为未完成)
    jobs = []
    for job_name, data_file in found:
        job_dir = MSA_OUTPUT_DIR / job_name
        marker_files = list(job_dir.rglob(f"{job_name}_model.cif")) if job_dir.exists() else []
        if skip_existing and any(m.stat().st_size > 0 for m in marker_files):
            print(f"[skip] {job_name} already has complete local results")
            continue
        jobs.append((job_name, data_file))

    if not jobs:
        print("Nothing to do.")
        return

    print("=" * 60)
    print(f"Found {len(found)} MSA folder(s), {len(jobs)} to run inference")
    print("=" * 60)

    # --- 上传本地 data.json 到 volume (按 {job}/{job}_data.json 落位) ---
    print("\n[Upload] Uploading local data.json to volume...")
    for job_name, data_file in jobs:
        if volume_has_msa_cache(job_name):
            print(f"  [SKIP] {job_name:20s} already in volume")
            continue
        upload_file_to_volume(msa_cache_volume, data_file, msa_remote_path(job_name))
        print(f"  [OK]   {job_name:20s} uploaded {data_file.name}")

    # --- 查 volume 上已完成的结果: 跳过推理直接下载 (省钱 + 断点续跑) ---
    try:
        result_entries = list(results_volume.iterdir("/", recursive=True))
    except (FileNotFoundError, modal.exception.NotFoundError):
        result_entries = []
    done_files = {}                          # job 名 -> 该 job 在 volume 上的文件名集合
    for e in result_entries:
        if e.type == modal.volume.FileEntryType.FILE:
            job, _, rest = e.path.partition("/")
            if rest and (getattr(e, 'size', 0) or 0) > 0:
                done_files.setdefault(job, set()).add(pathlib.Path(e.path).name)

    def volume_has_result(job_name: str) -> bool:
        """volume 上有非空的 {job}_model.cif 才算完成 (只认目录名会把崩溃留下的空目录当成功)"""
        return f"{job_name}_model.cif" in done_files.get(job_name, set())

    to_infer = [(n, p) for n, p in jobs if not volume_has_result(n)]
    reused = len(jobs) - len(to_infer)
    if reused:
        print(f"[reuse] volume 上已有 {reused} 个完成的结果, 跳过推理直接下载")

    # --- 跑 inference ---
    if to_infer:
        print(f"Running inference for {len(to_infer)} job(s)...")
        print("=" * 60)
        inf_args = [(job_name,) for job_name, _ in to_infer]
        list(run_inference.starmap(inf_args, order_outputs=True))
    else:
        print("所有任务在 volume 上都已有结果, 无需推理")

    # 有新推理的就重列一次(整体一次调用), 然后按 job 分组供下载直接使用
    if to_infer:
        try:
            result_entries = list(results_volume.iterdir("/", recursive=True))
        except (FileNotFoundError, modal.exception.NotFoundError):
            result_entries = []
    by_job = {}
    for e in result_entries:
        by_job.setdefault(e.path.partition("/")[0], []).append(e)

    # --- 下载推理结果到本地 ---
    print(f"\n[Download Results] -> {MSA_OUTPUT_DIR}")
    MSA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        futures = {
            pool.submit(
                download_from_volume,
                results_volume,
                job_name,
                MSA_OUTPUT_DIR / job_name,
                True,          # skip_bulky: 不下 *_confidences.json 与 *_data.json
                by_job.get(job_name),   # 预列好的条目, 不再每个 job 列一次目录
            ): job_name
            for job_name, _ in jobs
        }
        for fut in concurrent.futures.as_completed(futures):
            job_name = futures[fut]
            try:
                n = fut.result()
                if n == 0:                   # 0 个文件说明没真正下到东西, 不能当成功
                    print(f"  [EMPTY] {job_name:20s} 0 files —— 结果未下载!")
                else:
                    print(f"  [OK]   {job_name:20s} {n} files")
            except Exception as e:
                print(f"  [FAIL] {job_name:20s} download failed: {e}")

    print("Inference done.")


# ============================================================
# 入口 4: MSA-free 推理 (不跑 data pipeline, 直接用原始序列推理)
# 用法: modal run af3_modal.py::only_inference_no_msa
# 精度会明显下降,适用于快速筛查/孤儿蛋白/de novo 设计蛋白
# ============================================================
@app.local_entrypoint()
def only_inference_no_msa(skip_existing: bool = True):
    """
    从 INPUT_DIR 读原始序列 JSON, 不跑 data pipeline, 直接 MSA-free 推理:
      1. 扫描 INPUT_DIR 下所有 .json
      2. 本地把每个 JSON "加工" (补齐空 MSA/templates/modifications) 后保存到 NO_MSA_DIR
      3. 每个 job 跑 run_inference_no_msa
      4. 下载推理结果到本地 NO_MSA_OUTPUT_DIR

    skip_existing: NO_MSA_OUTPUT_DIR 下已存在结果目录的 job 跳过 (默认 True)
    """
    import concurrent.futures

    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    json_files = sorted(INPUT_DIR.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {INPUT_DIR}")

    NO_MSA_DIR.mkdir(parents=True, exist_ok=True)

    # 以存在非空 {job}_model.cif 为完成标志, 0 字节或缺失都视为未完成
    jobs = []
    for jf in json_files:
        job_name = jf.stem
        job_dir = NO_MSA_OUTPUT_DIR / job_name
        marker_files = list(job_dir.rglob(f"{job_name}_model.cif")) if job_dir.exists() else []
        if skip_existing and any(m.stat().st_size > 0 for m in marker_files):
            print(f"[skip] {job_name} already has complete local results")
            continue
        raw = jf.read_text(encoding="utf-8")
        transformed = transform_to_msa_free(raw)
        (NO_MSA_DIR / f"{job_name}.json").write_text(transformed, encoding="utf-8")
        jobs.append((job_name, transformed))

    if not jobs:
        print("Nothing to do.")
        return

    print("=" * 60)
    print(f"Found {len(json_files)} input(s), {len(jobs)} to run MSA-free inference")
    print(f"Input dir:    {INPUT_DIR}")
    print(f"Processed:    {NO_MSA_DIR}")
    print(f"Output dir:   {NO_MSA_OUTPUT_DIR}")
    print("=" * 60)
    print("NOTE: Precision will be significantly lower than MSA-based inference.")
    print("=" * 60)

    print(f"\nRunning MSA-free inference for {len(jobs)} job(s)...")
    args = [(job_name, raw_json) for job_name, raw_json in jobs]
    list(run_inference_no_msa.starmap(args, order_outputs=True))

    print(f"\n[Download Results] -> {NO_MSA_OUTPUT_DIR}")
    NO_MSA_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(
                download_from_volume,
                results_volume,
                job_name,
                NO_MSA_OUTPUT_DIR / job_name,
                True,          # skip_bulky: 不下 *_confidences.json 与 *_data.json
            ): job_name
            for job_name, _ in jobs
        }
        for fut in concurrent.futures.as_completed(futures):
            job_name = futures[fut]
            try:
                n = fut.result()
                print(f"  [OK]   {job_name:20s} {n} files")
            except Exception as e:
                print(f"  [FAIL] {job_name:20s} download failed: {e}")

    print("MSA-free inference done.")
