# FYP
Final Year Project

### Data Preprocess
1. Use `download_data.py` to select a dataset and download it. The downloaded datasets will be in the "XXX.npz" format.
2. The "XXX.npz" format is designed for NumPy. You can use `preprocess.ipynb` to convert the file to a uniform `csv` format.
3. Benchmark: `RandomForest.py` will train its model using the generated dataset, which will take about 5 minutes. Given that `RandomForest.py` uses `scikit-learn`, which is a state-of-the-art ML library, the C training model that I will write later will definitely be slower than that.
