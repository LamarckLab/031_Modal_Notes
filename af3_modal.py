import os
import pathlib
import modal

app = modal.App("alphafold3-batch")

# ============================================================
# 本地路径配置
# ============================================================
INPUT_DIR = pathlib.Path(r"C:\Users\Lamarck\Desktop\af3_inputs")              # 输入 JSON
MSA_DIR = pathlib.Path(r"C:\Users\Lamarck\Desktop\af3_msa")                   # 本地 MSA 缓存
MSA_OUTPUT_DIR = pathlib.Path(r"C:\Users\Lamarck\Desktop\af3_msa_outputs")        # MSA-based 推理结果
NO_MSA_DIR = pathlib.Path(r"C:\Users\Lamarck\Desktop\af3_no_msa")                 # MSA-free 加工后的 JSON
NO_MSA_OUTPUT_DIR = pathlib.Path(r"C:\Users\Lamarck\Desktop\af3_no_msa_outputs")  # MSA-free 推理结果


# ============================================================
# 严格对照 AlphaFold3 官方 Dockerfile 构建 af3_image
# ============================================================
af3_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-base-ubuntu24.04",
        add_python="3.12",
    )
    .apt_install(
        "git", "wget",
        "gcc", "g++", "make",
        "zlib1g-dev", "zstd",
        "patch", "clang",
    )
    .pip_install("uv==0.9.24")
    .env({
        "UV_COMPILE_BYTECODE": "1",
        "UV_PROJECT_ENVIRONMENT": "/alphafold3_venv",
        "PATH": "/hmmer/bin:/alphafold3_venv/bin:/usr/local/bin:/usr/bin:/bin",
    })
    .run_commands("uv venv /alphafold3_venv")
    .run_commands(
        "git clone https://github.com/google-deepmind/alphafold3.git /app/alphafold",
    )
    .run_commands(
        "mkdir -p /hmmer_build /hmmer",
        "wget http://eddylab.org/software/hmmer/hmmer-3.4.tar.gz -P /hmmer_build",
        "cd /hmmer_build && echo 'ca70d94fd0cf271bd7063423aabb116d42de533117343a9b27a65c17ff06fbf3  hmmer-3.4.tar.gz' | sha256sum --check",
        "cd /hmmer_build && tar zxf hmmer-3.4.tar.gz && rm hmmer-3.4.tar.gz",
    )
    .run_commands(
        "cp /app/alphafold/docker/jackhmmer_seq_limit.patch /hmmer_build/",
        "cd /hmmer_build && patch -p0 < jackhmmer_seq_limit.patch",
    )
    .run_commands(
        "cd /hmmer_build/hmmer-3.4 && ./configure --prefix=/hmmer && make -j4",
        "cd /hmmer_build/hmmer-3.4 && make install",
        "cd /hmmer_build/hmmer-3.4/easel && make install",
        "rm -rf /hmmer_build",
    )
    .run_commands(
        "cd /app/alphafold && UV_HTTP_TIMEOUT=1800 uv sync --frozen --all-groups --no-editable",
    )
    .run_commands(
        "cd /app/alphafold && uv run build_data",
    )
    .env({
        "XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "true",
        "XLA_CLIENT_MEM_FRACTION": "0.95",
    })
)


# ============================================================
# Volumes: 数据库/权重 + MSA 缓存 + 推理结果
# ============================================================
af3_volume = modal.Volume.from_name("alphafold3-data")

msa_cache_volume = modal.Volume.from_name(
    "alphafold3-msa-cache",
    create_if_missing=True,
)

results_volume = modal.Volume.from_name(
    "alphafold3-results",
    create_if_missing=True,
)


# ============================================================
# 缓存路径约定 (本地和 volume 结构一致)
#   子文件夹: {job_name}-msa-cache
#   缓存文件: {job_name}-msa-cache.json
# ============================================================
CACHE_SUFFIX = "-msa-cache"


def cache_dir_name(job_name: str) -> str:
    return f"{job_name}{CACHE_SUFFIX}"


def cache_file_name(job_name: str) -> str:
    return f"{job_name}{CACHE_SUFFIX}.json"


# ============================================================
# 函数 1: 数据管线阶段 (MSA + 模板搜索)
# 产出: /msa_cache/{job_name}-msa-cache/{job_name}-msa-cache.json
# ============================================================
@app.function(
    image=af3_image,
    volumes={
        "/data": af3_volume,
        "/msa_cache": msa_cache_volume,
    },
    cpu=24,
    memory=16384,
    timeout=60 * 60 * 12,
)
def run_data_pipeline(fasta_json: str, job_name: str) -> str:
    import subprocess
    import pathlib
    import shutil
    import os

    cache_subdir = cache_dir_name(job_name)
    cache_file = cache_file_name(job_name)
    target_dir = pathlib.Path(f"/msa_cache/{cache_subdir}")
    target_file = target_dir / cache_file

    msa_cache_volume.reload()
    if target_file.exists():
        print(f"[cache hit] job={job_name}")
        return job_name

    print(f"[cache miss] job={job_name}, running data pipeline...")

    # 输入 JSON 写到容器内临时路径
    input_dir = pathlib.Path("/tmp/af_input")
    input_dir.mkdir(parents=True, exist_ok=True)
    input_path = input_dir / f"{job_name}.json"
    input_path.write_text(fasta_json)

    # AF3 先写到容器内临时输出目录,跑完后再重命名到 volume
    tmp_out = pathlib.Path(f"/tmp/af_out/{job_name}")
    if tmp_out.exists():
        shutil.rmtree(tmp_out)
    tmp_out.mkdir(parents=True, exist_ok=True)

    cmd = [
        "uv", "run", "python3", "/app/alphafold/run_alphafold.py",
        f"--json_path={input_path}",
        "--db_dir=/data/databases",
        f"--output_dir={tmp_out}",
        "--norun_inference",
        "--jackhmmer_n_cpu=6",
    ]
    subprocess.run(cmd, check=True, cwd="/app/alphafold")

    # 找到 AF3 产出的 *_data.json,按约定重命名并放到 volume
    data_jsons = list(tmp_out.rglob("*_data.json"))
    if not data_jsons:
        raise FileNotFoundError(f"No *_data.json produced by AF3 in {tmp_out}")
    source = data_jsons[0]

    target_dir.mkdir(parents=True, exist_ok=True)
    # 分块写入 + fsync 强制刷盘, 避免 shutil.copy2 在 Modal volume FUSE 挂载上
    # 因 sparse-write / page cache 未回写导致的文件损坏
    with open(source, "rb") as src_f, open(target_file, "wb") as dst_f:
        shutil.copyfileobj(src_f, dst_f, length=1024 * 1024)
        dst_f.flush()
        os.fsync(dst_f.fileno())

    src_size = source.stat().st_size
    dst_size = target_file.stat().st_size
    if src_size != dst_size:
        raise RuntimeError(
            f"MSA cache write size mismatch for {job_name}: "
            f"src={src_size} dst={dst_size}"
        )

    shutil.rmtree(tmp_out, ignore_errors=True)
    msa_cache_volume.commit()
    print(f"[done] MSA cached at {target_file}")
    return job_name


# ============================================================
# 函数 2: 推理阶段
# 读取: /msa_cache/{job_name}-msa-cache/{job_name}-msa-cache.json
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
)
def run_inference(job_name: str) -> str:
    import subprocess
    import pathlib
    import shutil

    msa_cache_volume.reload()

    cache_subdir = cache_dir_name(job_name)
    cache_file = cache_file_name(job_name)
    data_json_path = pathlib.Path(f"/msa_cache/{cache_subdir}/{cache_file}")

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

    cmd = [
        "uv", "run", "python3", "/app/alphafold/run_alphafold.py",
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
        "uv", "run", "python3", "/app/alphafold/run_alphafold.py",
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
def download_from_volume(volume: modal.Volume, remote_prefix: str, local_dir: pathlib.Path) -> int:
    """把 volume 下 remote_prefix 目录递归下载到 local_dir

    - 每个文件先写到 .part, 完整且大小匹配后再 rename 到正式名
    - 单文件失败不影响其他文件, 打日志继续
    - 写入后 fsync, 避免 OS 级缓存未刷盘
    - 返回成功下载的文件数
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    prefix = remote_prefix.rstrip("/")
    try:
        entries = list(volume.iterdir(f"{prefix}/", recursive=True))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return 0

    success = 0
    failed = []
    for entry in entries:
        if entry.type != modal.volume.FileEntryType.FILE:
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

    if failed:
        print(f"  [WARN] {len(failed)} file(s) failed under prefix '{prefix}'")
    return success


def upload_dir_to_volume(volume: modal.Volume, local_dir: pathlib.Path, remote_prefix: str) -> int:
    """把 local_dir 上传到 volume 的 remote_prefix 路径 (覆盖)"""
    file_count = sum(1 for p in local_dir.rglob("*") if p.is_file())
    if file_count == 0:
        return 0
    with volume.batch_upload(force=True) as batch:
        batch.put_directory(str(local_dir), remote_prefix)
    return file_count


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
    cache_subdir = cache_dir_name(job_name)
    cache_file = cache_file_name(job_name)
    target_path = f"{cache_subdir}/{cache_file}"
    try:
        for entry in msa_cache_volume.iterdir(f"{cache_subdir}/", recursive=True):
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
                cache_dir_name(job_name),
                MSA_DIR / cache_dir_name(job_name),
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
                cache_dir_name(job_name),
                MSA_DIR / cache_dir_name(job_name),
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
    扫描本地 MSA_DIR 下所有 {job_name}-msa-cache 子文件夹:
      1. 把本地缓存上传到 volume (volume 已有则跳过上传)
      2. 跑 inference
      3. 下载推理结果到本地 MSA_OUTPUT_DIR

    skip_existing: 本地已存在结果目录的 job 跳过 (默认 True)
    """
    import concurrent.futures

    if not MSA_DIR.exists():
        raise FileNotFoundError(f"Local MSA cache dir not found: {MSA_DIR}")

    # 扫描 -msa-cache 结尾的子文件夹
    cache_folders = []
    for d in sorted(MSA_DIR.iterdir()):
        if not d.is_dir():
            continue
        if not d.name.endswith(CACHE_SUFFIX):
            print(f"[skip] {d.name} 文件夹名不以 '{CACHE_SUFFIX}' 结尾")
            continue
        job_name = d.name[:-len(CACHE_SUFFIX)]
        expected_file = d / cache_file_name(job_name)
        if not expected_file.exists():
            print(f"[skip] {d.name} 缺失 {expected_file.name}")
            continue
        cache_folders.append((job_name, d))

    if not cache_folders:
        print(f"No valid cache folders in {MSA_DIR}")
        return

    # 过滤已有本地结果的 (以存在非空 {job}_model.cif 为完成标志, 0 字节或缺失都视为未完成)
    jobs = []
    for job_name, cache_folder in cache_folders:
        job_dir = MSA_OUTPUT_DIR / job_name
        marker_files = list(job_dir.rglob(f"{job_name}_model.cif")) if job_dir.exists() else []
        if skip_existing and any(m.stat().st_size > 0 for m in marker_files):
            print(f"[skip] {job_name} already has complete local results")
            continue
        jobs.append((job_name, cache_folder))

    if not jobs:
        print("Nothing to do.")
        return

    print("=" * 60)
    print(f"Found {len(cache_folders)} cache(s), {len(jobs)} to run inference")
    print("=" * 60)

    # --- 上传本地缓存到 volume ---
    print("\n[Upload] Uploading local cache(s) to volume...")
    for job_name, cache_folder in jobs:
        if volume_has_msa_cache(job_name):
            print(f"  [SKIP] {job_name:20s} already in volume")
            continue
        n = upload_dir_to_volume(
            msa_cache_volume,
            cache_folder,
            cache_dir_name(job_name),
        )
        print(f"  [OK]   {job_name:20s} uploaded {n} file(s)")

    # --- 跑 inference ---
    print(f"\nRunning inference for {len(jobs)} job(s)...")
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
