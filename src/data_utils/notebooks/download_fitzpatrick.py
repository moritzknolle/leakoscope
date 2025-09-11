import pandas as pd
import requests
from pathlib import Path
from tqdm import tqdm


DATA_DIR = Path("/home/moritz/data/fitzpatrick17k")
CSV_PATH = "/home/moritz/repositories/fair_leak/data/csv/fitzpatrick17k.csv"

# Load CSV file
df = pd.read_csv(CSV_PATH)
df["download_success"] = False

# Directory to save the files
DATA_DIR.mkdir(exist_ok=True)


# Function to download file
def download_file(url, filename, directory):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3",
        "Accept": "*/*",  # You can adjust this as needed
    }
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Check for HTTP errors

        filename = directory / filename
        with open(filename, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {url}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")
        return False


# Download each file
for url, md5 in tqdm(zip(df["url"], df["md5hash"]), total=len(df)):
    if not Path(DATA_DIR / f"{md5}.jpg").exists():
        success = download_file(
            url=url,
            filename=f"{md5}.jpg",
            directory=DATA_DIR,
        )
        df.loc[df["md5hash"] == md5, "download_success"] = success
    else:
        df.loc[df["md5hash"] == md5, "download_success"] = True

df.to_csv(CSV_PATH, index=False)
