import pandas as pd

def test_basic_dataframe():
    df = pd.DataFrame({"Close": [100, 101, 102]})
    assert "Close" in df.columns
