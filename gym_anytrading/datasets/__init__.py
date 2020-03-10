from .utils import load_dataset as _load_dataset


# Load FOREX datasets
FOREX_EURUSD_1H_ASK = _load_dataset('FOREX_EURUSD_1H_ASK', 'Time')

# Load Stocks datasets
STOCKS_GOOGL = _load_dataset('STOCKS_GOOGL', 'Date')

# Load XAU datasets
XAUUSD_1T = _load_dataset('XAUUSD_1T', 'Time')
XAUUSD_1H = _load_dataset('XAUUSD_1H', 'Time')
XAUUSD_4H = _load_dataset('XAUUSD_4H', 'Time')
XAUUSD_1D = _load_dataset('XAUUSD_1D', 'Time')

# Load Forex datasets
EURUSD_1T = _load_dataset('EURUSD_1T', 'Time')
EURUSD_1H = _load_dataset('EURUSD_1H', 'Time')
EURUSD_4H = _load_dataset('EURUSD_4H', 'Time')
EURUSD_1D = _load_dataset('EURUSD_1D', 'Time')


