"""Part 1: one-step-ahead temperature forecasting with a GRU.

Loads the Jena climate temperature series (10-minute readings,
2009-2016), holds out the last 1,440 observations (10 days) as the test
window, trains GRU(50) on lookback-60 windows of the min-max-scaled
series, and evaluates once on the test window against a persistent
forecast computed on the identical 1,440 targets. Writes
part-1_output.txt and jena_predictions_gru.png next to this script;
downloads the CSV on first run if it is missing.
"""

import io
import os
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers, models
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "jena_climate_2009_2016.csv"
DATA_URL = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    "jena_climate_2009_2016.csv.zip"
)

TEST_SIZE = 1440  # last 10 days of 10-minute readings
LOOKBACK = 60  # 10 hours of context per prediction
MAX_EPOCHS = 15
BATCH_SIZE = 64
SEED = 42


def load_temperature_series() -> np.ndarray:
    """Return T (degC) as a float array; fetch the CSV next to this script if absent."""
    if not CSV_PATH.exists():
        print(f"Downloading Jena climate dataset to {CSV_PATH} ...")
        with urllib.request.urlopen(DATA_URL) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        with archive.open(CSV_PATH.name) as source:
            CSV_PATH.write_bytes(source.read())
    return pd.read_csv(CSV_PATH)["T (degC)"].to_numpy(dtype="float64")


def make_windows(series: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    """Pair each lookback-length window with the observation that follows it."""
    windows = np.lib.stride_tricks.sliding_window_view(series, lookback)[:-1]
    targets = series[lookback:]
    return np.ascontiguousarray(windows)[..., np.newaxis], targets


def build_gru_model(lookback: int) -> keras.Model:
    model = models.Sequential(
        [
            keras.Input(shape=(lookback, 1)),
            layers.GRU(50),
            layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model


def metric_row(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> str:
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return f"{name:<24} {mse:>10.4f} {np.sqrt(mse):>10.4f} {mae:>10.4f}"


def plot_predictions(actual: np.ndarray, predicted: np.ndarray, save_path: Path) -> None:
    plt.figure(figsize=(15, 6))
    plt.plot(actual, color="blue", label="Actual temperature", linewidth=1.5)
    plt.plot(
        predicted,
        color="red",
        linestyle="--",
        label="GRU one-step-ahead prediction",
        linewidth=1.5,
        alpha=0.8,
    )
    plt.title("Jena temperature, last 10 days of test data: GRU vs actual")
    plt.xlabel("Time (10-minute intervals)")
    plt.ylabel("Temperature (\N{DEGREE SIGN}C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def versions_header() -> str:
    import sklearn
    import tensorflow as tf

    return "\n".join(
        [
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Versions: keras {keras.__version__}, tensorflow {tf.__version__}, "
            f"scikit-learn {sklearn.__version__}",
            f"Random seed: {SEED}",
        ]
    )


def main() -> None:
    keras.utils.set_random_seed(SEED)

    series = load_temperature_series()
    train_raw, test_raw = series[:-TEST_SIZE], series[-TEST_SIZE:]

    # Scale on the training data only; the test window is transformed
    # with the same fitted scaler (no leakage).
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_scaled = scaler.fit_transform(train_raw.reshape(-1, 1)).ravel()
    test_scaled = scaler.transform(test_raw.reshape(-1, 1)).ravel()

    x_train, y_train = make_windows(train_scaled, LOOKBACK)

    # Prefix the test window with the last LOOKBACK training points so
    # every one of the 1,440 test observations gets a prediction.
    x_test, _ = make_windows(
        np.concatenate([train_scaled[-LOOKBACK:], test_scaled]), LOOKBACK
    )

    model = build_gru_model(LOOKBACK)
    history = model.fit(
        x_train,
        y_train,
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.1,  # Keras takes the tail slice: the latest data
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        ],
        verbose=0,
    )
    epochs_run = len(history.history["loss"])
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1

    predicted = scaler.inverse_transform(model.predict(x_test, verbose=0)).ravel()

    # Persistent forecast on the identical targets: each prediction is
    # the previous observation (the last training point for the first).
    persistent = np.concatenate([train_raw[-1:], test_raw[:-1]])

    plot_predictions(test_raw, predicted, HERE / "jena_predictions_gru.png")

    gru_mse = mean_squared_error(test_raw, predicted)
    persistent_mse = mean_squared_error(test_raw, persistent)
    header = f"{'Model':<24} {'MSE':>10} {'RMSE':>10} {'MAE':>10}"
    report = "\n".join(
        [
            "PART 1 - GRU TEMPERATURE FORECASTING VS PERSISTENCE",
            "=" * 60,
            versions_header(),
            "",
            "Configuration:",
            f"  Series: T (degC), {len(series)} observations at 10-minute steps",
            f"  Test window: last {TEST_SIZE} observations (10 days)",
            f"  Lookback: {LOOKBACK} steps (10 hours), one-step-ahead targets",
            f"  Model: GRU(50) -> Dense(1), {model.count_params()} parameters, adam",
            f"  Training: max {MAX_EPOCHS} epochs, batch {BATCH_SIZE}, val split 0.1 "
            "(temporal tail),",
            "  early stopping on val loss (patience 3, best weights restored)",
            f"  Epochs run: {epochs_run}, best epoch: {best_epoch}",
            "",
            "Test window, single evaluation (units: \N{DEGREE SIGN}C):",
            header,
            "-" * len(header),
            metric_row("GRU one-step-ahead", test_raw, predicted),
            metric_row("Persistent forecast", test_raw, persistent),
            "",
            f"MSE improvement over persistence: "
            f"{(persistent_mse - gru_mse) / persistent_mse * 100:.1f}%",
            "",
        ]
    )
    (HERE / "part-1_output.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
