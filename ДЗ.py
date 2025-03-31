import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score


def analyze_model_parameters(params):
    """Анализирует параметры модели и выводит текстовый комментарий"""
    print("\nАнализ параметров модели:")

    ar, ma = params['ar.L1'], params['ma.L1']
    sar, sma = params['ar.S.L7'], params['ma.S.L7']

    # Анализ AR компоненты
    if abs(ar) < 0.1:
        print(f"- AR компонента ({ar:.3f}): слабая, возможно можно исключить")
    else:
        print(f"- AR компонента ({ar:.3f}): значимая, сохраняет память о предыдущих значениях")

    # Анализ MA компоненты
    if abs(ma) > 0.7:
        print(f"- MA компонента ({ma:.3f}): сильная, хорошо учитывает шоки")
    else:
        print(f"- MA компонента ({ma:.3f}): умеренная")

    # Анализ сезонных компонент
    if abs(sar) < 0.1:
        print(f"- Сезонная AR ({sar:.3f}): слабая, возможно избыточна")
    else:
        print(f"- Сезонная AR ({sar:.3f}): значимая")

    if abs(sma) > 0.9:
        print(f"- Сезонная MA ({sma:.3f}): очень сильная, четкая недельная сезонность")
    else:
        print(f"- Сезонная MA ({sma:.3f}): умеренная сезонность")


def main():
    # Загрузка данных
    train_path = r'C:\Users\hhh\Desktop\Настя работа\train.csv'
    test_path = r'C:\Users\hhh\Desktop\Настя работа\test.csv'

    try:
        train_df = pd.read_csv(train_path, parse_dates=['Date'], index_col='Date')
        test_df = pd.read_csv(test_path, parse_dates=['Date'], index_col='Date')
        print("Данные успешно загружены!")
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")
        return

    # 1. Анализ разных уровней агрегации
    print("\nАнализ разных уровней агрегации:")
    for freq in ['D', 'W', 'ME']:  # Исправлено на 'ME' вместо 'M'
        resampled = train_df['number_sold'].resample(freq).mean()
        print(f"{freq}: {len(resampled)} точек, среднее: {resampled.mean():.1f}")

    # 2. Анализ по магазинам и товарам
    print("\nАнализ по магазинам:")
    store_stats = train_df.groupby('store')['number_sold'].agg(['mean', 'std', 'count'])
    print(store_stats)

    print("\nАнализ по товарам:")
    product_stats = train_df.groupby('product')['number_sold'].agg(['mean', 'std', 'count'])
    print(product_stats)

    # Основное моделирование
    print("\nОсновное моделирование:")
    daily_train = train_df['number_sold'].resample('D').mean().ffill()
    daily_test = test_df['number_sold'].resample('D').mean().ffill()

    model = SARIMAX(daily_train,
                    order=(1, 1, 1),
                    seasonal_order=(1, 1, 1, 7),
                    enforce_stationarity=False)

    model_fit = model.fit(disp=False)
    print(model_fit.summary())

    # Анализ параметров модели
    analyze_model_parameters(model_fit.params)

    # Прогнозирование и оценка
    forecast = model_fit.get_forecast(steps=len(daily_test))
    forecast_values = forecast.predicted_mean

    # Метрики качества
    mape = mean_absolute_percentage_error(daily_test, forecast_values)
    rmse = np.sqrt(mean_squared_error(daily_test, forecast_values))
    r2 = r2_score(daily_test, forecast_values)

    print("\nОкончательные метрики качества:")
    print(f"MAPE: {mape:.2%}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2: {r2:.4f}")

    # Комментарий по качеству модели
    print("\nОценка качества модели:")
    if mape < 0.01:
        print("Отличное качество модели (MAPE < 1%)")
    elif mape < 0.05:
        print("Хорошее качество модели (MAPE < 5%)")
    else:
        print("Удовлетворительное качество модели")

    if r2 > 0.9:
        print("Модель объясняет более 90% дисперсии данных (R2 > 0.9)")
    elif r2 > 0.7:
        print("Модель объясняет более 70% дисперсии данных")

    # Анализ остатков
    residuals = model_fit.resid
    print("\nАнализ остатков модели:")
    print(f"Среднее остатков: {residuals.mean():.4f}")
    print(f"Стандартное отклонение: {residuals.std():.4f}")
    print(f"Тест Дики-Фуллера для остатков (p-value): {adfuller(residuals.dropna())[1]:.4f}")


if __name__ == "__main__":
    main()