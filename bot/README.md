# 🤖 Vollautomatischer Trading-Bot für Binance

Ein fortschrittlicher Trading-Bot mit KI-gestützter Analyse, Sentiment-Analyse und umfassendem Risikomanagement für Binance.

## 🚀 Features

### Core Trading Engine
- **Realtime-Kurse** via Binance WebSocket
- **Technische Indikatoren**: RSI, MACD, Bollinger Bands, EMA-Cross, ATR
- **Intelligente Entry/Exit-Regeln** basierend auf Multi-Signal-Analyse
- **Positions-Sizing**: Max. 2% Kontokapital pro Trade
- **Dynamisches Risikomanagement** mit ATR-basierten Stop-Loss/Take-Profit

### KI-Modul
- **LSTM & Random Forest** Modelle für Trading-Signale
- **Automatisches Training** mit historischen Daten
- **Feature Engineering** mit technischen Indikatoren
- **Confidence-basierte** Entscheidungsfindung

### News & Sentiment Layer
- **Crypto-News** von CoinDesk und CryptoPanic
- **Sentiment-Analyse** mit OpenAI und HuggingFace
- **Dynamische Positionsanpassung** basierend auf Marktstimmung

### Web Dashboard
- **Streamlit-basiertes** Dashboard
- **Live-Kurse** und Equity-Curve
- **Offene Positionen** und Trading-Historie
- **Einstellungs-Panel** für alle Parameter

### Telegram-Benachrichtigungen
- **Echtzeit-Alerts** für Trades, SL/TP-Hits
- **Tägliche Zusammenfassungen**
- **Fehler-Benachrichtigungen**

### Backtesting & Paper Trading
- **Historische Tests** mit Backtrader
- **Paper Trading** für risikofreies Testen
- **Performance-Metriken** (Sharpe Ratio, Max Drawdown, etc.)

## 📋 Voraussetzungen

- Python 3.8+
- Binance Account mit API-Keys
- Telegram Bot (optional)
- OpenAI API Key (optional für Sentiment-Analyse)

## 🛠️ Installation

### 1. Repository klonen
```bash
git clone <repository-url>
cd trading-bot
```

### 2. Dependencies installieren
```bash
pip install -r requirements.txt
```

### 3. Umgebungsvariablen konfigurieren
```bash
# env_example.txt zu .env kopieren
cp env_example.txt .env

# API-Keys in .env eintragen
nano .env
```

### 4. Konfiguration anpassen
```bash
# config.yaml nach Bedarf anpassen
nano config.yaml
```

## ⚙️ Konfiguration

### API-Keys (.env)
```env
# Binance API Keys
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here

# Telegram Bot
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# OpenAI API (für Sentiment-Analyse)
OPENAI_API_KEY=your_openai_api_key_here

# CryptoPanic API (für News)
CRYPTOPANIC_API_KEY=your_cryptopanic_api_key_here
```

### Trading-Parameter (config.yaml)
```yaml
trading:
  mode: "paper"  # paper, live, backtest
  symbol: "BTCUSDT"
  timeframe: "1h"
  max_position_size: 0.02  # 2% des Kontos
  max_positions: 3

risk:
  stop_loss_atr_multiplier: 2.0
  take_profit_atr_multiplier: 3.0
  max_daily_loss: 0.05  # 5% max täglicher Verlust
  trailing_stop: true
```

## 🚀 Verwendung

### 1. Paper Trading starten
```bash
python main.py --mode paper
```

### 2. Dashboard öffnen
```bash
python main.py --mode dashboard
```
Das Dashboard ist dann unter `http://localhost:8501` verfügbar.

### 3. Backtest ausführen
```bash
python main.py --mode backtest --start-date 2023-01-01 --end-date 2024-01-01
```

### 4. AI-Modell trainieren
```bash
python main.py --mode train
```

### 5. Verbindungen testen
```bash
python main.py --mode test
```

### 6. Live Trading (VORSICHT!)
```bash
python main.py --mode live
```

## 📊 Dashboard Features

### Hauptansicht
- **Live-Status** des Bots
- **Account-Balance** und P&L
- **Offene Positionen** und Trade-Historie
- **Performance-Metriken**

### Charts
- **Preis-Chart** mit technischen Indikatoren
- **Performance-Chart** mit Equity-Curve
- **Trade-Tabelle** mit detaillierten Informationen
- **Markt-Analyse** mit Signal-Breakdown

### Kontrollen
- **Bot Start/Stop**
- **Trading-Modus** ändern
- **Risiko-Parameter** anpassen
- **Symbol und Timeframe** konfigurieren

## 🔧 Trading-Strategie

### Entry-Signale
- **RSI < 30** (Oversold) + Preis über EMA200
- **MACD Bullish Crossover**
- **Bollinger Bands** Support
- **AI-Modell** Confidence > 70%
- **Positives Sentiment** (optional)

### Exit-Signale
- **RSI > 70** (Overbought)
- **MACD Bearish Crossover**
- **Stop-Loss** oder **Take-Profit** Hit
- **Trend-Reversal** Erkennung
- **AI-Modell** Sell-Signal

### Risikomanagement
- **Position Sizing**: 1% Risiko pro Trade
- **Stop-Loss**: 2x ATR unter Entry
- **Take-Profit**: 3x ATR über Entry
- **Trailing Stop**: 1% Abstand
- **Max. 3 Positionen** gleichzeitig
- **Täglicher Verlust-Limit**: 5%

## 📈 Performance-Metriken

### Backtest-Ergebnisse
- **Total Return**: Gesamtrendite
- **Sharpe Ratio**: Risiko-adjustierte Rendite
- **Max Drawdown**: Maximaler Verlust
- **Win Rate**: Gewinnrate
- **Average Trade**: Durchschnittlicher Trade
- **Best/Worst Trade**: Bester/schlechtester Trade

## 🔒 Sicherheit

### Paper Trading
- **Keine echten Trades** im Paper-Modus
- **Simulierte Ausführung** mit echten Marktdaten
- **Risikofreies Testen** der Strategie

### Live Trading
- **API-Keys** sicher speichern
- **Kleine Positionen** zum Start
- **Stop-Loss** immer aktiviert
- **Regelmäßige Überwachung**

## 🧪 Tests

### Unit Tests ausführen
```bash
pytest tests/
```

### Spezifische Tests
```bash
# Indicator Tests
pytest tests/test_indicators.py

# Strategy Tests
pytest tests/test_strategy.py

# Risk Management Tests
pytest tests/test_risk_manager.py
```

## 📝 Logging

### Log-Dateien
- `trades.log`: Alle Trading-Entscheidungen
- `trading_bot.log`: Bot-System-Logs

### Log-Level
- **INFO**: Normale Operationen
- **WARNING**: Warnungen
- **ERROR**: Fehler
- **DEBUG**: Detaillierte Debug-Informationen

## 🐛 Troubleshooting

### Häufige Probleme

#### 1. Binance API Fehler
```
Error: Binance API error
```
**Lösung**: API-Keys überprüfen, IP-Whitelist aktivieren

#### 2. Telegram Verbindungsfehler
```
Error: Telegram connection failed
```
**Lösung**: Bot-Token und Chat-ID überprüfen

#### 3. Keine Marktdaten
```
Error: No historical data available
```
**Lösung**: Internetverbindung prüfen, Symbol überprüfen

#### 4. AI-Modell Fehler
```
Error: AI model not available
```
**Lösung**: TensorFlow/Scikit-learn installieren, Modell trainieren

## 📚 API-Dokumentation

### Hauptklassen

#### Config
```python
from src.config import Config
config = Config("config.yaml")
trading_config = config.get_trading_config()
```

#### DataManager
```python
from src.data_manager import DataManager
data_manager = DataManager(config)
df = data_manager.get_historical_data("BTCUSDT", "1h")
```

#### Strategy
```python
from src.strategy import Strategy
strategy = Strategy(config, indicator_engine, ai_model, news_analyzer)
analysis = strategy.analyze_market(df)
```

#### Trader
```python
from src.trader import Trader
trader = Trader(config, data_manager, strategy, risk_manager, telegram_notifier)
trader.start()
```

## 🤝 Beitragen

1. **Fork** das Repository
2. **Feature Branch** erstellen (`git checkout -b feature/AmazingFeature`)
3. **Commit** die Änderungen (`git commit -m 'Add some AmazingFeature'`)
4. **Push** zum Branch (`git push origin feature/AmazingFeature`)
5. **Pull Request** erstellen

## 📄 Lizenz

Dieses Projekt ist unter der MIT-Lizenz lizenziert - siehe [LICENSE](LICENSE) Datei für Details.

## ⚠️ Haftungsausschluss

**WICHTIG**: Dieser Trading-Bot ist für Bildungszwecke gedacht. Trading mit Kryptowährungen ist hochriskant und kann zu erheblichen Verlusten führen. Verwenden Sie nur Geld, das Sie sich leisten können zu verlieren.

- **Keine Finanzberatung**: Dies ist keine Finanzberatung
- **Eigene Verantwortung**: Sie handeln auf eigene Verantwortung
- **Risiko**: Kryptowährungen sind volatil und riskant
- **Testen**: Immer zuerst im Paper-Trading-Modus testen

## 📞 Support

Bei Fragen oder Problemen:
- **Issues** auf GitHub erstellen
- **Dokumentation** durchsuchen
- **Logs** überprüfen

---

**Viel Erfolg beim Trading! 🚀📈** 