import requests
import pandas as pd
from datetime import datetime, timedelta

def get_economic_calendar():
    """
    Simulates a monitor for high-impact economic events.
    In production, this would hit an API like AlphaVantage or FinancialModelingPrep.
    """
    # High-impact events for the coming week (simulated)
    events = [
        {"event": "FOMC Statement", "date": "2025-02-21 19:00:00", "impact": "High"},
        {"event": "Non-Farm Payrolls", "date": "2025-03-07 13:30:00", "impact": "High"}
    ]
    
    now = datetime.utcnow()
    for e in events:
        event_time = datetime.strptime(e['date'], "%Y-%m-%d %H:%M:%S")
        # Check if we are in the 30min window
        if abs((now - event_time).total_seconds()) < 1800:
            print(f"!!! NEWS-GAP FILTER: {e['event']} DETECTED. PAUSING ENTRIES !!!")
            return False
    return True

print("Initializing News-Gap Filter (Sentiment Phase 3)...")
get_economic_calendar()
