"""ATLAS universe, currency map, asset classes, bond durations."""
from __future__ import annotations

STOCKS = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
    "GOOGL": "Alphabet A", "GOOG": "Alphabet C", "META": "Meta Platforms",
    "BRK-B": "Berkshire Hathaway B", "LLY": "Eli Lilly", "AVGO": "Broadcom",
    "JPM": "JPMorgan Chase", "V": "Visa", "MA": "Mastercard", "XOM": "Exxon Mobil",
    "COST": "Costco", "WMT": "Walmart", "NFLX": "Netflix", "ORCL": "Oracle",
    "AMD": "AMD", "CRM": "Salesforce", "ADBE": "Adobe", "QCOM": "Qualcomm",
    "INTC": "Intel", "TSLA": "Tesla", "KO": "Coca-Cola", "PEP": "PepsiCo",
    "UNH": "UnitedHealth", "CAT": "Caterpillar", "GE": "GE Aerospace", "IBM": "IBM",
    "CSCO": "Cisco", "AMAT": "Applied Materials", "MU": "Micron", "PLTR": "Palantir",
    "UBER": "Uber", "ABNB": "Airbnb", "DIS": "Walt Disney", "MCD": "McDonald's",
    "BA": "Boeing", "GS": "Goldman Sachs", "BAC": "Bank of America", "C": "Citigroup",
    "WFC": "Wells Fargo", "CVX": "Chevron", "COP": "ConocoPhillips", "PFE": "Pfizer",
    "MRK": "Merck", "TMO": "Thermo Fisher", "ISRG": "Intuitive Surgical", "NOW": "ServiceNow",
    "PANW": "Palo Alto Networks", "SNPS": "Synopsys", "CRWD": "CrowdStrike", "SHOP": "Shopify",
    "DELL": "Dell Technologies", "ARM": "Arm Holdings", "SMCI": "Super Micro Computer",
    "LRCX": "Lam Research", "KLAC": "KLA", "TXN": "Texas Instruments", "BKNG": "Booking Holdings",
    "SPOT": "Spotify", "PYPL": "PayPal", "T": "AT&T", "VZ": "Verizon", "NKE": "Nike",
    "LOW": "Lowe's", "HD": "Home Depot", "UPS": "UPS", "FDX": "FedEx", "AXP": "American Express",
    "BLK": "BlackRock", "SCHW": "Charles Schwab", "COIN": "Coinbase", "MSTR": "Strategy",
    "F": "Ford", "GM": "General Motors",
    "SAP": "SAP", "SIE.DE": "Siemens", "ALV.DE": "Allianz", "DTE.DE": "Deutsche Telekom",
    "MUV2.DE": "Munich Re", "AIR.PA": "Airbus", "MC.PA": "LVMH", "OR.PA": "L'Oreal",
    "TTE.PA": "TotalEnergies", "SAN.PA": "Sanofi", "SU.PA": "Schneider Electric", "ASML": "ASML",
    "RHM.DE": "Rheinmetall", "MBG.DE": "Mercedes-Benz", "BMW.DE": "BMW", "VOW3.DE": "Volkswagen",
    "BAS.DE": "BASF", "IFX.DE": "Infineon", "ADS.DE": "Adidas", "DBK.DE": "Deutsche Bank",
    "ENR.DE": "Siemens Energy", "BN.PA": "Danone", "NOVO-B.CO": "Novo Nordisk", "NVO": "Novo Nordisk ADR",
    "AZN": "AstraZeneca", "SHEL": "Shell", "HSBA.L": "HSBC", "BP.L": "BP", "ULVR.L": "Unilever",
    "RIO": "Rio Tinto", "GSK": "GSK", "REL.L": "RELX", "LSEG.L": "London Stock Exchange Group",
    "BHP": "BHP Group", "NESN.SW": "Nestle", "ROG.SW": "Roche", "NOVN.SW": "Novartis",
    "UBSG.SW": "UBS", "7203.T": "Toyota", "6758.T": "Sony", "9984.T": "SoftBank Group",
    "005930.KS": "Samsung Electronics", "000660.KS": "SK Hynix", "0700.HK": "Tencent",
    "9988.HK": "Alibaba", "3690.HK": "Meituan", "TSM": "TSMC ADR", "BABA": "Alibaba ADR",
    "INFY": "Infosys ADR", "RELIANCE.NS": "Reliance Industries",
    "ACN": "Accenture", "AMGN": "Amgen", "BX": "Blackstone", "KKR": "KKR",
    "SNOW": "Snowflake", "DDOG": "Datadog", "LMT": "Lockheed Martin", "RTX": "RTX",
    "NOC": "Northrop Grumman", "GD": "General Dynamics",
}

ETFS = {
    "SPY": "SPDR S&P 500", "VOO": "Vanguard S&P 500", "QQQ": "Invesco Nasdaq-100",
    "VTI": "Vanguard Total US Market", "VT": "Vanguard Total World", "ACWI": "iShares MSCI ACWI",
    "VEA": "Vanguard Developed Markets", "VWO": "Vanguard Emerging Markets",
    "IEFA": "iShares Core MSCI EAFE", "IEMG": "iShares Core MSCI Emerging Markets",
    "EFA": "iShares MSCI EAFE", "EEM": "iShares MSCI Emerging Markets",
    "DIA": "SPDR Dow Jones", "IWM": "iShares Russell 2000", "SCHD": "Schwab US Dividend",
    "VIG": "Vanguard Dividend Appreciation", "XLK": "Technology Select Sector",
    "XLF": "Financial Select Sector", "XLE": "Energy Select Sector", "XLV": "Health Care Select Sector",
    "XLI": "Industrial Select Sector", "SMH": "VanEck Semiconductor", "SOXX": "iShares Semiconductor",
    "ARKK": "ARK Innovation", "TLT": "iShares 20+ Year Treasury", "IEF": "iShares 7-10 Year Treasury",
    "SHY": "iShares 1-3 Year Treasury", "AGG": "iShares Core US Aggregate Bond",
    "BND": "Vanguard Total Bond Market", "LQD": "iShares IG Corporate Bond",
    "HYG": "iShares High Yield Corporate Bond", "TIP": "iShares TIPS Bond",
    "GLD": "SPDR Gold Shares", "IAU": "iShares Gold Trust", "SLV": "iShares Silver Trust",
    "PDBC": "Invesco Commodity", "DBC": "Invesco DB Commodity", "USO": "US Oil Fund",
    "UNG": "US Natural Gas Fund", "VNQ": "Vanguard Real Estate", "REET": "iShares Global REIT",
    "EUNL.DE": "iShares Core MSCI World UCITS", "IWDA.AS": "iShares Core MSCI World",
    "VWCE.DE": "Vanguard FTSE All-World UCITS", "VUSA.AS": "Vanguard S&P 500 UCITS",
    "CSPX.L": "iShares Core S&P 500 UCITS", "SXR8.DE": "iShares Core S&P 500 UCITS",
    "EXSA.DE": "iShares STOXX Europe 600", "IS3N.DE": "iShares Core MSCI EM IMI",
}

COMMODITIES = {
    "GC=F": "Gold futures", "SI=F": "Silver futures", "PL=F": "Platinum futures",
    "PA=F": "Palladium futures", "HG=F": "Copper futures", "CL=F": "WTI crude oil",
    "BZ=F": "Brent crude oil", "NG=F": "Natural gas", "ALI=F": "Aluminium futures",
    "ZC=F": "Corn futures", "ZW=F": "Wheat futures", "ZS=F": "Soybean futures",
    "KC=F": "Coffee futures", "SB=F": "Sugar futures", "CT=F": "Cotton futures",
    "CC=F": "Cocoa futures",
}

BOND_DURATION = {
    "TLT": 16.5, "IEF": 7.5, "SHY": 1.9, "AGG": 6.1, "BND": 6.2,
    "LQD": 8.2, "HYG": 3.6, "TIP": 6.8, "GOVT": 6.0, "VGIT": 5.0,
}

CURRENCY_MAP = {
    "USD": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "BRK-B", "LLY", "AVGO",
            "JPM", "V", "MA", "XOM", "COST", "WMT", "NFLX", "ORCL", "AMD", "CRM", "ADBE",
            "QCOM", "INTC", "TSLA", "KO", "PEP", "UNH", "CAT", "GE", "IBM", "CSCO", "AMAT",
            "MU", "PLTR", "UBER", "ABNB", "DIS", "MCD", "BA", "GS", "BAC", "C", "WFC",
            "CVX", "COP", "PFE", "MRK", "TMO", "ISRG", "NOW", "PANW", "SNPS", "CRWD",
            "SHOP", "DELL", "ARM", "SMCI", "LRCX", "KLAC", "TXN", "BKNG", "SPOT", "PYPL",
            "T", "VZ", "NKE", "LOW", "HD", "UPS", "FDX", "AXP", "BLK", "SCHW", "COIN",
            "MSTR", "F", "GM", "SPY", "VOO", "QQQ", "VTI", "VT", "ACWI", "VEA", "VWO",
            "IEFA", "IEMG", "EFA", "EEM", "DIA", "IWM", "SCHD", "VIG", "XLK", "XLF",
            "XLE", "XLV", "XLI", "SMH", "SOXX", "ARKK", "TLT", "IEF", "SHY", "AGG",
            "BND", "LQD", "HYG", "TIP", "GLD", "IAU", "SLV", "PDBC", "DBC", "USO",
            "UNG", "VNQ", "REET", "TSM", "BABA", "NVO", "ASML", "SHEL", "AZN", "GSK",
            "RIO", "BHP", "ACN", "AMGN", "BX", "KKR", "SNOW", "DDOG", "LMT", "RTX",
            "NOC", "GD", "GC=F", "SI=F", "PL=F", "PA=F", "HG=F", "CL=F", "BZ=F", "NG=F"],
    "EUR": ["SAP", "SIE.DE", "ALV.DE", "DTE.DE", "MUV2.DE", "AIR.PA", "MC.PA", "OR.PA",
            "TTE.PA", "SAN.PA", "SU.PA", "RHM.DE", "MBG.DE", "BMW.DE", "VOW3.DE",
            "BAS.DE", "IFX.DE", "ADS.DE", "DBK.DE", "ENR.DE", "BN.PA", "EUNL.DE",
            "VWCE.DE", "SXR8.DE", "EXSA.DE", "IS3N.DE"],
    "GBP": ["HSBA.L", "BP.L", "ULVR.L", "REL.L", "LSEG.L", "CSPX.L"],
    "CHF": ["NESN.SW", "ROG.SW", "NOVN.SW", "UBSG.SW"],
    "JPY": ["7203.T", "6758.T", "9984.T"],
    "KRW": ["005930.KS", "000660.KS"],
    "HKD": ["0700.HK", "9988.HK", "3690.HK"],
    "INR": ["RELIANCE.NS", "INFY"],
    "DKK": ["NOVO-B.CO"],
}

def ticker_currency(ticker: str) -> str:
    for ccy, tickers in CURRENCY_MAP.items():
        if ticker in tickers:
            return ccy
    if ticker.endswith((".DE", ".PA", ".AS")):
        return "EUR"
    if ticker.endswith(".L"):
        return "GBP"
    if ticker.endswith(".SW"):
        return "CHF"
    if ticker.endswith(".T"):
        return "JPY"
    if ticker.endswith(".KS"):
        return "KRW"
    if ticker.endswith(".HK"):
        return "HKD"
    if ticker.endswith(".NS"):
        return "INR"
    return "USD"

ASSET_CLASS = {}
for t in STOCKS:
    ASSET_CLASS[t] = "Equity"
for t in ETFS:
    if t in BOND_DURATION:
        ASSET_CLASS[t] = "Fixed Income"
    elif t in ("GLD", "IAU", "SLV", "PDBC", "DBC", "USO", "UNG"):
        ASSET_CLASS[t] = "Commodity"
    elif t in ("VNQ", "REET"):
        ASSET_CLASS[t] = "Real Estate"
    else:
        ASSET_CLASS[t] = "Equity ETF"
for t in COMMODITIES:
    ASSET_CLASS[t] = "Commodity"

ALL_UNIVERSE = {**STOCKS, **ETFS, **COMMODITIES}
