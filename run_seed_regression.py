# -*- coding: utf-8 -*-
"""
run_seed_regression.py
G 组回归(G1/G2): 换 seed 生成新视图集 -> 全链路重建 -> 一键评估。
一条命令可重复执行: 生成视图 -> 重建 -> 评估。
同时校验 A1: 不同 seed 的重建输出必须互不相同(管线消费上传图, 无自渲染旁路)。

用法:
  python run_seed_regression.py               # 默认 seed 1,2
  python run_seed_regression.py --seeds 1,2,3
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

OUTDIR = "evaluation"


def run(cmd):
    r = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', default='1,2')
    args = ap.parse_args()
    seeds = [int(x) for x in args.seeds.split(',') if x != '']

    os.makedirs(OUTDIR, exist_ok=True)
    from web_core_genviews import generate_views_from_matlab
    from web_core_images import run_images_pipeline

    all_ok = True
    digest_by_seed = {}
    results = []
    for s in seeds:
        vdir = os.path.join(OUTDIR, f"views_s{s}")
        rdir = os.path.join(OUTDIR, f"recon_s{s}")
        generate_views_from_matlab("plantroot06.m", vdir, seed=s)
        pose = json.load(open(os.path.join(vdir, "pose.json"), encoding="utf-8"))
        res = run_images_pipeline(os.path.join(vdir, "views"), rdir, pose=pose)
        print(f"[seed {s}] 重建:", res['stats'])

        h = hashlib.md5(open(res['points_table'], 'rb').read()).hexdigest()
        digest_by_seed[s] = h

        cmd = [sys.executable, "evaluate_reconstruction.py",
               "--seed", str(s),
               "--points", res['points_table'],
               "--controls", res['control_table'],
               "--root-csv", os.path.join(rdir, "root_table.csv"),
               "--root-sql", os.path.join(rdir, "root_table.sql"),
               "--detections", os.path.join(rdir, "detections.json"),
               "--label", f"seed{s}", "--outdir", OUTDIR]
        rc = run(cmd)
        ok = (rc == 0)
        all_ok = all_ok and ok
        results.append((s, ok))

    # A1: 不同 seed 的重建输出必须互不相同
    digests = list(digest_by_seed.values())
    a1_ok = len(set(digests)) == len(digests)
    print("\n" + "=" * 62)
    print(f"A1 输出随上传变化(不同 seed 输出互不相同): {'通过' if a1_ok else '失败'}")
    for s, ok in results:
        print(f"  seed {s}: 完成标准 {'全部通过' if ok else '存在未通过项'} "
              f"(详见 {OUTDIR}/eval_seed{s}_seed{s}.md)")
    print("=" * 62)
    if not a1_ok:
        all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
