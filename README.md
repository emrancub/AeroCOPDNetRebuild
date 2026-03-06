# AeroCOPDNetRebuild (Reproducible Major-Revision Pipeline)

This repository is a clean, from-scratch rebuild for the AeroCOPDNet COPD-from-lung-sounds project.
It focuses on **reproducibility**, **subject-wise splitting**, and **reviewer-required experiments**.

## 0) Setup

### Create venv (Windows)
```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Create venv (Linux/Mac)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Dataset paths (your machine)

You told us:

- ICBHI root:
  `F:\COPD Research\Respiratory Sound Database`

- Fraiwan root:
  `F:\COPD Research\jwyy9np4gv-3`

**Important:** Paths with spaces should be quoted.

## 2) Build manifests (subject-wise ready)

This creates:
- `manifests/icbhi_binary.csv`
- `manifests/fraiwan_binary.csv`
- `manifests/pooled_binary.csv`
- diagnosis breakdown tables in `manifests/`

```bat
python scripts\build_manifest.py ^
  --icbhi_root "F:\COPD Research\Respiratory Sound Database" ^
  --fraiwan_root "F:\COPD Research\jwyy9np4gv-3" ^
  --out_dir manifests ^
  --icbhi_audio_subdir "audio_and_txt_files" ^
  --icbhi_diag_csv "patient_diagnosis.csv" ^
  --fraiwan_audio_subdir "Audio Files" ^
  --fraiwan_anno_xlsx "Data annotation.xlsx" ^
  --fraiwan_window_sec 5.0 ^
  --fraiwan_hop_sec 2.5
```

Notes:
- **ICBHI**: uses cycle segments from the per-recording `.txt` annotation files (subject-wise).
- **Fraiwan**: if cycle annotations are not available, we default to **fixed windows** per recording
  (configurable via `--fraiwan_window_sec` / `--fraiwan_hop_sec`).

## 3) Run pooled 5-fold subject-wise CV (AeroCOPDNet)

```bat
python scripts\run_cv.py --config configs\cv_pooled_aerocpdnet.yaml
```

Outputs are created under:
`outputs/<experiment_name>/<timestamp>/fold_*/...`

## 4) Run baselines (CNN / CRNN / LSTM / GRU / MobileNetV2)

```bat
python scripts\run_cv.py --config configs\cv_pooled_baselines.yaml
python scripts\run_cv.py --config configs\mobilenet_comparison.yaml
```

## 5) Augmentation ablation (No Aug / +SpecAug / +Mixup / +SpecAug+Mixup)

```bat
python scripts\run_ablation_aug.py --config configs\ablation_aug.yaml
```

## 6) Architecture ablation (Depthwise / SE / Pooling head)

```bat
python scripts\run_ablation_arch.py --config configs\ablation_arch.yaml
```

## 7) Cross-dataset generalization (Train on one, test on the other)

```bat
python scripts\run_cross_dataset.py --config configs\cross_train_icbhi_test_fraiwan.yaml
python scripts\run_cross_dataset.py --config configs\cross_train_fraiwan_test_icbhi.yaml
```

## 8) Summarize results into paper tables + generate all figures at 1000 DPI

```bat
python scripts\summarize_results.py --outputs_dir outputs --out_dir paper_assets
python scripts\make_all_figures.py --outputs_dir outputs --out_dir paper_assets --dpi 1000
```

---

## 📚 Citation

If you use this paper, please cite our manuscript:

```bibtex
@article{hasan2026aerocopdnet,
  title={AeroCOPDNet: A deep learning framework for COPD detection from lung sounds},
  author={Hasan, Md Emran and Wu, Yue-Fang and Yu, Dong-Jun},
  journal={Biomedical Signal Processing and Control},
  volume={119},
  pages={109939},
  year={2026},
  publisher={Elsevier}
}
```

---

## 📄 License

Add a license file (e.g., **MIT**) in `LICENSE`.
Respect the original dataset licenses and citation requirements.

---

## 📬 Contact

* For research questions, email the corresponding author listed in the paper.
* Or, Please contact **Md Emran Hasan** ([writetoemran@gmail.com](mailto:writetoemran@gmail.com) or [mdemranhasan@njust.edu.cn](mailto:mdemranhasan@njust.edu.cn)).
