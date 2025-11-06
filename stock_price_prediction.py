import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import yfinance as yf
from datetime import datetime, timedelta

# 1. DATA COLLECTION
def fetch_stock_data(ticker, period='2y'):
    """Fetch stock data from Yahoo Finance"""
    print(f"Fetching data for {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    return df

# 2. FEATURE ENGINEERING
def create_features(df):
    """Create technical indicators as features"""
    df = df.copy()
    
    # Simple Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    
    # Exponential Moving Average
    df['EMA_12'] = df['Close'].ewm(span=12).mean()
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
    
    # Price Change Percentage
    df['Price_Change'] = df['Close'].pct_change()
    
    # Volume Change
    df['Volume_Change'] = df['Volume'].pct_change()
    
    # Create target: 1 if next day price goes up, 0 if down
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df

# 3. PREPARE DATA FOR ML MODEL
def prepare_data(df):
    """Prepare features and target for training"""
    feature_columns = ['SMA_5', 'SMA_20', 'EMA_12', 'RSI', 'MACD', 
                      'Price_Change', 'Volume_Change']
    
    X = df[feature_columns]
    y = df['Target']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )
    
    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# 4. TRAIN MODEL
def train_model(X_train, y_train):
    """Train Random Forest Classifier"""
    print("Training model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

# 5. EVALUATE MODEL
def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    print(f"\n{'='*50}")
    print("MODEL PERFORMANCE")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, 
                                target_names=['Down', 'Up']))
    
    return predictions, accuracy

# 6. TRADING SIMULATION
def simulate_trading(df, predictions, initial_capital=10000):
    """Simulate simple trading strategy"""
    df_test = df.iloc[-len(predictions):].copy()
    df_test['Prediction'] = predictions
    
    capital = initial_capital
    shares = 0
    trade_log = []
    
    for i in range(len(df_test) - 1):
        current_price = df_test.iloc[i]['Close']
        prediction = df_test.iloc[i]['Prediction']
        
        # Buy signal: prediction is UP and we don't own shares
        if prediction == 1 and shares == 0:
            shares = capital / current_price
            capital = 0
            trade_log.append(('BUY', current_price, shares))
        
        # Sell signal: prediction is DOWN and we own shares
        elif prediction == 0 and shares > 0:
            capital = shares * current_price
            trade_log.append(('SELL', current_price, capital))
            shares = 0
    
    # Final liquidation
    if shares > 0:
        final_price = df_test.iloc[-1]['Close']
        capital = shares * final_price
        trade_log.append(('SELL', final_price, capital))
    
    profit = capital - initial_capital
    return_pct = (profit / initial_capital) * 100
    
    print(f"\n{'='*50}")
    print("TRADING SIMULATION RESULTS")
    print(f"{'='*50}")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Capital: ${capital:,.2f}")
    print(f"Profit/Loss: ${profit:,.2f}")
    print(f"Return: {return_pct:.2f}%")
    print(f"\nNumber of trades: {len(trade_log)}")
    
    return capital, trade_log

# 7. VISUALIZATION
def plot_results(df, predictions):
    """Plot stock price and predictions"""
    df_test = df.iloc[-len(predictions):].copy()
    df_test['Prediction'] = predictions
    
    plt.figure(figsize=(14, 7))
    
    # Plot stock price
    plt.subplot(2, 1, 1)
    plt.plot(df_test.index, df_test['Close'], label='Close Price', linewidth=2)
    
    # Mark buy signals
    buy_signals = df_test[df_test['Prediction'] == 1]
    plt.scatter(buy_signals.index, buy_signals['Close'], 
               color='green', marker='^', s=100, label='Buy Signal', alpha=0.7)
    
    # Mark sell signals
    sell_signals = df_test[df_test['Prediction'] == 0]
    plt.scatter(sell_signals.index, sell_signals['Close'], 
               color='red', marker='v', s=100, label='Sell Signal', alpha=0.7)
    
    plt.title('Stock Price with Buy/Sell Signals', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot RSI
    plt.subplot(2, 1, 2)
    plt.plot(df_test.index, df_test['RSI'], label='RSI', color='purple', linewidth=2)
    plt.axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
    plt.axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
    plt.title('Relative Strength Index (RSI)', fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('RSI')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# MAIN EXECUTION
def main():
    # Configuration
    TICKER = 'AAPL'  # Change to any stock ticker (AAPL, GOOGL, MSFT, etc.)
    INITIAL_CAPITAL = 10000
    
    print(f"\n{'='*50}")
    print(f"AI/ML STOCK TRADING PROJECT")
    print(f"{'='*50}\n")
    
    # Step 1: Fetch data
    df = fetch_stock_data(TICKER)
    
    # Step 2: Create features
    df = create_features(df)
    print(f"Dataset shape: {df.shape}")
    
    # Step 3: Prepare data
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)
    
    # Step 4: Train model
    model = train_model(X_train, y_train)
    
    # Step 5: Evaluate model
    predictions, accuracy = evaluate_model(model, X_test, y_test)
    
    # Step 6: Simulate trading
    final_capital, trade_log = simulate_trading(df, predictions, INITIAL_CAPITAL)
    
    # Step 7: Visualize results
    plot_results(df, predictions)
    
    print(f"\n{'='*50}")
    print("Analysis Complete!")
    print(f"{'='*50}\n")

if __name__ == "__main__":
    main()