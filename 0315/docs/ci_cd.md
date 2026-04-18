# CI/CD & GitHub Actions 指南

專案使用 GitHub Actions 實現 CI/CD，確保程式碼品質與自動化部署。

## GitHub Actions Workflows

### CI Pipeline (`.github/workflows/ci.yml`)
觸發條件：Push/PR 到 main/master 分支

**Jobs:**
1. **lint** - Ruff + Black + MyPy 檢查
2. **test** - 執行 pytest 測試，上傳覆蓋率報告
3. **smoke-test** - 驗證模組匯入和基本功能

### CD Pipeline (`.github/workflows/cd.yml`)
觸發條件：Push 到 main/master 分支（app.py 或 src/ 變更）

**功能:**
- 自動部署到 Hugging Face Spaces
- 需要設定 `HF_TOKEN` 和 `HF_USERNAME` secrets

### Retraining Pipeline (`.github/workflows/retraining.yml`)
觸發條件：每週六 01:30 UTC 或手動觸發

**功能:**
- 執行 Feature Pipeline 抓取最新資料
- 執行 Training Pipeline 重新訓練模型
- 需要設定 `HOPSWORKS_PROJECT` 和 `HOPSWORKS_API_KEY` secrets

## 設定 GitHub Secrets

在 GitHub repository Settings → Secrets and variables → Actions 加入：

| Secret 名稱 | 說明 |
|------------|------|
| `HOPSWORKS_PROJECT` | Hopsworks 專案名稱 |
| `HOPSWORKS_API_KEY` | Hopsworks API Key |
| `HF_TOKEN` | Hugging Face API Token（CD 用）|
| `HF_USERNAME` | Hugging Face 使用者名稱（CD 用）|
