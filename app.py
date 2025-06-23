from flask import Flask, render_template_string, request, send_file, url_for
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import io
import os

app = Flask(__name__)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stock Forecast</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: white;
        }

        .header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            text-align: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        }

        .header h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }

        .container {
            display: flex;
            max-width: 1400px;
            margin: 2rem auto;
            gap: 2rem;
            padding: 0 2rem;
        }

        .sidebar {
            width: 350px;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            height: fit-content;
        }

        .sidebar h2 {
            margin-bottom: 2rem;
            color: #fff;
            font-size: 1.5rem;
        }

        .form-group {
            margin-bottom: 2rem;
        }

        .form-group label {
            display: block;
            margin-bottom: 0.5rem;
            font-weight: 600;
            color: rgba(255, 255, 255, 0.9);
        }

        select, input {
            width: 100%;
            padding: 0.75rem;
            border: none;
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.9);
            color: #333;
            font-size: 1rem;
        }

        button {
            width: 100%;
            padding: 1rem;
            background: linear-gradient(45deg, #ff6b6b, #ee5a6f);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            font-size: 1rem;
            transition: all 0.3s ease;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(255, 107, 107, 0.4);
        }

        .model-info {
            margin-top: 2rem;
            padding: 1.5rem;
            background: rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        .model-info h3 {
            margin-bottom: 1rem;
            color: #fff;
        }

        .main-content {
            flex: 1;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-radius: 15px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            text-align: center;
        }

        .main-content h1 {
            margin-bottom: 2rem;
            color: #fff;
            font-size: 2rem;
        }

        .chart-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 10px;
            padding: 1rem;
            margin-bottom: 2rem;
        }

        img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }

        .download-link {
            display: inline-block;
            padding: 1rem 2rem;
            background: linear-gradient(45deg, #00d4ff, #0099cc);
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.3s ease;
        }

        .download-link:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4);
        }

        @media (max-width: 768px) {
            .container {
                flex-direction: column;
            }
            .sidebar {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>📈 Stock Price Forecast</h1>
        <p>Advanced AI-powered stock prediction using multiple machine learning models</p>
    </div>

    <div class="container">
        <div class="sidebar">
            <h2>🎯 Forecast Settings</h2>
            <form method="get" id="modelForm">
                <div class="form-group">
                    <label for="model">Select Model:</label>
                    <select id="model" name="model" onchange="document.getElementById('modelForm').submit()">
                        <option value="arima" {% if model == 'arima' %}selected{% endif %}>ARIMA (Auto-Regressive)</option>
                        <option value="prophet" {% if model == 'prophet' %}selected{% endif %}>Prophet (Facebook AI)</option>
                        <option value="ema" {% if model == 'ema' %}selected{% endif %}>EMA (Moving Average)</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="months">Forecast Period (months):</label>
                    <input type="number" id="months" name="months" min="1" max="24" value="{{ months }}" onchange="document.getElementById('modelForm').submit()"/>
                </div>
                
                <button type="submit">🚀 Generate Forecast</button>
            </form>
            
            <div class="model-info">
                {% if model == 'arima' %}
                    <h3>About ARIMA</h3>
                    <p>ARIMA (AutoRegressive Integrated Moving Average) is a classic statistical model for time series forecasting, effective for stationary data with trends.</p>
                {% elif model == 'prophet' %}
                    <h3>About Prophet</h3>
                    <p>Prophet is a modern forecasting tool developed by Facebook, designed for easy modeling of seasonality and trends in time series data.</p>
                {% elif model == 'ema' %}
                    <h3>About EMA</h3>
                    <p>EMA (Exponential Moving Average) smooths the data giving more weight to recent points. This simple EMA forecast repeats the last EMA value for future points.</p>
                {% endif %}
            </div>
        </div>
        
        <div class="main-content">
            <h1>Stock Price Analysis</h1>
            <div class="chart-container">
                <img src="{{ url_for('static', filename='forecast_plot.png') }}" alt="Forecast Plot">
            </div>
            <a class="download-link" href="{{ url_for('download_csv', model=model, months=months) }}">
                📥 Download Forecast CSV
            </a>
        </div>
    </div>
</body>
</html>
'''

def load_data():
    data = {
        'Date': [
            'Jul 1, 2023', 'Aug 1, 2023', 'Sep 1, 2023', 'Oct 1, 2023', 'Nov 1, 2023', 'Dec 1, 2023',
            'Jan 1, 2024', 'Feb 1, 2024', 'Mar 1, 2024', 'Apr 1, 2024', 'May 1, 2024', 'Jun 1, 2024',
            'Jul 1, 2024', 'Aug 1, 2024', 'Sep 1, 2024', 'Oct 1, 2024', 'Nov 1, 2024', 'Dec 1, 2024',
            'Jan 1, 2025', 'Feb 1, 2025', 'Mar 1, 2025', 'Apr 1, 2025', 'May 1, 2025', 'Jun 1, 2025', 'Jun 22, 2025'
        ],
        'Close': [
            1960.00, 1995.00, 2190.00, 1825.00, 2050.00, 1910.00,
            2267.00, 2520.00, 2567.00, 2479.00, 2550.00, 2761.00,
            3005.00, 3086.00, 3520.00, 3755.00, 4340.00, 4224.00,
            4062.00, 4804.00, 4364.00, 4467.00, 5784.00, 6137.00, 6600.00
        ]
    }
    df = pd.DataFrame(data)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df = df.sort_index()
    return df

df = load_data()

def ensure_static_dir():
    if not os.path.exists('static'):
        os.makedirs('static')

@app.route('/')
def index():
    model = request.args.get('model', 'arima').lower()
    try:
        months = int(request.args.get('months', 6))
    except ValueError:
        months = 6
    months = max(1, min(months, 24))

    ensure_static_dir()
    forecast_df = None

    if model == 'arima':
        arima_model = ARIMA(df['Close'], order=(3, 1, 2))
        model_fit = arima_model.fit()
        forecast = model_fit.forecast(steps=months)
        forecast_index = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(1), periods=months, freq='MS')
        forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast.values})
        forecast_df.set_index('Date', inplace=True)

        plt.figure(figsize=(12,6))
        plt.style.use('dark_background')
        plt.plot(df.index, df['Close'], label='Actual', marker='o', linewidth=2, color='#00d4ff')
        plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast (ARIMA)', linestyle='--', marker='x', color='#ff6b6b', linewidth=2)
        plt.title('Stock Price Forecast (ARIMA)', fontsize=16, color='white')
        plt.xlabel('Date', fontsize=12, color='white')
        plt.ylabel('Close Price ($)', fontsize=12, color='white')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('static/forecast_plot.png', facecolor='#1a1a2e', dpi=300, bbox_inches='tight')
        plt.close()

    elif model == 'prophet':
        prophet_df = df.reset_index().rename(columns={'Date':'ds', 'Close':'y'})
        m = Prophet()
        m.fit(prophet_df)

        future = m.make_future_dataframe(periods=months, freq='MS')
        forecast = m.predict(future)

        forecast_df = forecast[['ds', 'yhat']].tail(months).copy()
        forecast_df.set_index('ds', inplace=True)
        forecast_df.rename(columns={'yhat':'Forecast'}, inplace=True)

        plt.figure(figsize=(12,6))
        plt.style.use('dark_background')
        plt.plot(df.index, df['Close'], label='Actual', marker='o', linewidth=2, color='#00d4ff')
        plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast (Prophet)', linestyle='--', marker='x', color='#ff6b6b', linewidth=2)
        plt.title('Stock Price Forecast (Prophet)', fontsize=16, color='white')
        plt.xlabel('Date', fontsize=12, color='white')
        plt.ylabel('Close Price ($)', fontsize=12, color='white')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('static/forecast_plot.png', facecolor='#1a1a2e', dpi=300, bbox_inches='tight')
        plt.close()

    elif model == 'ema':
        span = 10
        ema_series = df['Close'].ewm(span=span, adjust=False).mean()

        last_ema = ema_series.iloc[-1]
        forecast_index = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(1), periods=months, freq='MS')
        forecast_values = [last_ema] * months
        forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast_values})
        forecast_df.set_index('Date', inplace=True)

        plt.figure(figsize=(12,6))
        plt.style.use('dark_background')
        plt.plot(df.index, df['Close'], label='Actual', marker='o', linewidth=2, color='#00d4ff')
        plt.plot(ema_series.index, ema_series.values, label=f'EMA (span={span})', color='#00ff88', linewidth=2)
        plt.plot(forecast_df.index, forecast_df['Forecast'], label='Forecast (EMA)', linestyle='--', marker='x', color='#ff6b6b', linewidth=2)
        plt.title('Stock Price Forecast (EMA)', fontsize=16, color='white')
        plt.xlabel('Date', fontsize=12, color='white')
        plt.ylabel('Close Price ($)', fontsize=12, color='white')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('static/forecast_plot.png', facecolor='#1a1a2e', dpi=300, bbox_inches='tight')
        plt.close()

    else:
        return "Model not supported", 400

    forecast_df.to_csv('static/forecast.csv')
    return render_template_string(HTML_TEMPLATE, model=model, months=months)

@app.route('/download')
def download_csv():
    model = request.args.get('model', 'arima').lower()
    try:
        months = int(request.args.get('months', 6))
    except ValueError:
        months = 6
    months = max(1, min(months, 24))

    if model == 'arima':
        arima_model = ARIMA(df['Close'], order=(3, 1, 2))
        model_fit = arima_model.fit()
        forecast = model_fit.forecast(steps=months)
        forecast_index = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(1), periods=months, freq='MS')
        forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast.values})
    elif model == 'prophet':
        prophet_df = df.reset_index().rename(columns={'Date':'ds', 'Close':'y'})
        m = Prophet()
        m.fit(prophet_df)
        future = m.make_future_dataframe(periods=months, freq='MS')
        forecast = m.predict(future)
        forecast_df = forecast[['ds', 'yhat']].tail(months).copy()
        forecast_df.rename(columns={'ds': 'Date', 'yhat':'Forecast'}, inplace=True)
    elif model == 'ema':
        span = 10
        ema_series = df['Close'].ewm(span=span, adjust=False).mean()
        last_ema = ema_series.iloc[-1]
        forecast_index = pd.date_range(start=df.index[-1] + pd.offsets.MonthBegin(1), periods=months, freq='MS')
        forecast_values = [last_ema] * months
        forecast_df = pd.DataFrame({'Date': forecast_index, 'Forecast': forecast_values})
    else:
        return "Model not supported", 400

    csv_io = io.StringIO()
    forecast_df.to_csv(csv_io, index=False)
    csv_io.seek(0)

    return send_file(
        io.BytesIO(csv_io.getvalue().encode()),
        mimetype='text/csv',
        as_attachment=True,
        download_name='forecast.csv'
    )

if __name__ == '__main__':
    app.run(debug=True, port=5009)
