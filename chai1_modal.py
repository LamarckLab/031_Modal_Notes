import os
import pathlib
import modal

app = modal.App("chai1-batch")

# ============================================================
# 本地路径配置
# ============================================================
INPUT_DIR = pathlib.Path(r"G:\Lab_Data\chai1_inputs")      # 输入 .fasta
OUTPUT_DIR = pathlib.Path(r"G:\Lab_Data\chai1_outputs")    # 推理结果

# ============================================================
# Chai-1 推理参数 (按需修改)
# 参考: https://github.com/chaidiscovery/chai-lab
# ============================================================
NUM_TRUNK_RECYCLES   = 3       # 主干循环次数, 越多越准但越慢
NUM_DIFFN_TIMESTEPS  = 200     # 扩散步数, 论文默认 200
NUM_DIFFN_SAMPLES    = 5       # 每个输入采样多少个结构 (会输出多个 cif)
SEED                 = 42
USE_ESM_EMBEDDINGS   = True    # 用 ESM2 单序列嵌入, Chai-1 默认 True
USE_MSA_SERVER       = True    # 直接打 ColabFold MMseqs2 公共服务器
MSA_SERVER_URL       = "https://api.colabfold.com"
USE_TEMPLATES_SERVER = False   # 是否用模板, 默认 False
LOW_MEMORY           = False   # 显存吃紧时设 True (会变慢)

# ============================================================
# Image: 用 PyTorch 官方 CUDA 镜像 + pip 装 chai_lab
# Chai-1 安装非常简单, 不需要 hmmer / 编译那一套
# ============================================================
chai1_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("git", "wget")
    .pip_install(
        "chai_lab==0.6.1",  # 按需锁版本; 想跟最新可以去掉 ==0.6.1
    )
    .env({
        # 把 chai_lab 的权重 / ESM / MSA 缓存目录指向 volume
        # chai_lab 用这个变量决定下载到哪儿
        "CHAI_DOWNLOADS_DIR": "/weights/chai-lab",
    })
)

# ============================================================
# Volumes:
#   chai1-weights : Chai-1 模型权重 + ESM2 缓存 (~10 GB, 持久化避免冷启动重下)
#   chai1-results : 推理结果
# ============================================================
weights_volume = modal.Volume.from_name(
    "chai1-weights",
    create_if_missing=True,
)

results_volume = modal.Volume.from_name(
    "chai1-results",
    create_if_missing=True,
)


# ============================================================
# 推理函数
# 输入: job_name + fasta 文本
# 产出: /results/{job_name}/  (含 N 个 *.cif + 评分 npz)
# ============================================================
@app.function(
    image=chai1_image,
    volumes={
        "/weights": weights_volume,
        "/results": results_volume,
    },
    gpu="H100",
    cpu=4,
    memory=16384,
    timeout=60 * 60 * 2,
)
def run_chai_inference(job_name: str, fasta_text: str) -> str:
    import pathlib
    import shutil
    from chai_lab.chai1 import run_inference  # type: ignore

    # 输入 FASTA 写到容器临时目录
    input_dir = pathlib.Path("/tmp/chai_input")
    input_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = input_dir / f"{job_name}.fasta"
    fasta_path.write_text(fasta_text)

    # 结果目录: /results/{job_name}
    result_dir = pathlib.Path(f"/results/{job_name}")
    if result_dir.exists():
        shutil.rmtree(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    # 同步一下权重 volume, 万一别的 worker 刚下载完
    weights_volume.reload()

    print(f"[{job_name}] running Chai-1 inference...")
    candidates = run_inference(
        fasta_file=fasta_path,
        output_dir=result_dir,
        num_trunk_recycles=NUM_TRUNK_RECYCLES,
        num_diffn_timesteps=NUM_DIFFN_TIMESTEPS,
        num_diffn_samples=NUM_DIFFN_SAMPLES,
        seed=SEED,
        device="cuda:0",
        use_esm_embeddings=USE_ESM_EMBEDDINGS,
        use_msa_server=USE_MSA_SERVER,
        msa_server_url=MSA_SERVER_URL,
        use_templates_server=USE_TEMPLATES_SERVER,
        low_memory=LOW_MEMORY,
    )

    # 首次跑会下载 ~10GB 权重到 /weights, commit 一下持久化
    weights_volume.commit()
    results_volume.commit()
    n_struct = len(getattr(candidates, "cif_paths", []) or [])
    print(f"[{job_name}] done, {n_struct} structures at /results/{job_name}")
    return str(result_dir)


# ============================================================
# 本地辅助: volume -> 本地 文件下载
# (复用 af3_modal.py 的稳健写法: .part 临时文件 + size 校验 + fsync)
# ============================================================
def download_from_volume(volume: modal.Volume, remote_prefix: str, local_dir: pathlib.Path) -> int:
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
            print(f"    [OK]   {entry.path} ({actual_size} bytes)")

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


# ============================================================
# 入口: 批量推理
# 用法: modal run chai1_modal.py
# ============================================================
@app.local_entrypoint()
def main(skip_existing: bool = True):
    """
    扫描 INPUT_DIR 下所有 .fasta / .fa:
      1. 本地已有完整结果的 job 跳过 (skip_existing=True)
      2. 上 GPU 跑 Chai-1 推理
      3. 下载结果到本地 OUTPUT_DIR
    """
    import concurrent.futures

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Input directory not found: {INPUT_DIR}\n"
            f"请在脚本顶部修改 INPUT_DIR, 或创建这个文件夹"
        )

    fasta_files = sorted(
        list(INPUT_DIR.glob("*.fasta")) + list(INPUT_DIR.glob("*.fa"))
    )
    if not fasta_files:
        raise FileNotFoundError(f"No .fasta/.fa files found in {INPUT_DIR}")

    jobs = []
    # 完成标志: OUTPUT_DIR/{job_name}/ 下至少一个非空 .cif
    for ff in fasta_files:
        job_name = ff.stem
        job_dir = OUTPUT_DIR / job_name
        cif_files = list(job_dir.rglob("*.cif")) if job_dir.exists() else []
        if skip_existing and any(c.stat().st_size > 0 for c in cif_files):
            print(f"[skip] {job_name} already has complete local results")
            continue
        jobs.append((job_name, ff.read_text(encoding="utf-8")))

    if not jobs:
        print("Nothing to do.")
        return

    print("=" * 60)
    print(f"Found {len(fasta_files)} input(s), {len(jobs)} to process")
    print(f"Input dir:  {INPUT_DIR}")
    print(f"Output dir: {OUTPUT_DIR}")
    print(
        f"Params: trunk={NUM_TRUNK_RECYCLES} "
        f"diffn_steps={NUM_DIFFN_TIMESTEPS} samples={NUM_DIFFN_SAMPLES} seed={SEED}"
    )
    print(
        f"MSA: server={USE_MSA_SERVER} "
        f"templates={USE_TEMPLATES_SERVER} ESM={USE_ESM_EMBEDDINGS}"
    )
    print("=" * 60)

    # --- 推理 ---
    print(f"\nRunning Chai-1 inference for {len(jobs)} job(s)...")
    list(run_chai_inference.starmap(jobs, order_outputs=True))

    # --- 下载结果到本地 ---
    print(f"\n[Download Results] -> {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {
            pool.submit(
                download_from_volume,
                results_volume,
                job_name,
                OUTPUT_DIR / job_name,
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
    print(f"  Results: {OUTPUT_DIR}")
    print("=" * 60)
