# 常見問題 (FAQ)

### Q: 如何新增特徵？
1. 在 `src/features.py` 新增計算邏輯
2. 在 `src/config.py` 的 `FEATURE_COLS` 加入特徵名稱
3. 新增測試案例
4. 執行 `python feature_pipeline.py && python training_pipeline.py`

### Q: 如何切換預測目標？
修改 `.env`：
```bash
TARGET_MODE=raw          # 預測原始報酬率
TARGET_MODE=excess_spy   # 預測超額報酬率（相對 SPY）
TARGET_HORIZON_DAYS=5    # 預測 N 日後收盤價
```

### Q: 本地測試不需要 Modal？
是的，每個 pipeline 都支援本地執行：
```bash
python feature_pipeline.py
python training_pipeline.py
python inference_pipeline.py
```

### Q: CI/CD 如何運作？
- **CI**：每次 push/PR 自動執行 lint + test
- **CD**：app.py 變更時自動部署到 HF Spaces
- **Retraining**：每週六自動執行重訓（GitHub Actions 備用）
