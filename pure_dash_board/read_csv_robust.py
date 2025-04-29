import pandas as pd
import chardet
import csv

# Detect the file encoding
with open("2015---Friuli-Venezia-Giulia---Gestione-finanziaria-Spese-Enti-Locali.csv", 'rb') as f:
    result = chardet.detect(f.read(10000))
    
print(f"Detected encoding: {result}")

# Try to detect the delimiter by reading a few lines
with open("2015---Friuli-Venezia-Giulia---Gestione-finanziaria-Spese-Enti-Locali.csv", 'r', encoding=result['encoding']) as f:
    sample = f.read(5000)
    sniffer = csv.Sniffer()
    try:
        dialect = sniffer.sniff(sample)
        print(f"Detected delimiter: '{dialect.delimiter}'")
        delimiter = dialect.delimiter
    except:
        print("Could not detect delimiter, trying common ones...")
        for delim in [',', ';', '\t', '|']:
            if delim in sample:
                print(f"Found '{delim}' in sample, using it as delimiter")
                delimiter = delim
                break
        else:
            print("No common delimiter found, defaulting to comma")
            delimiter = ','

# Try to read the CSV with the detected encoding and delimiter
try:
    df = pd.read_csv("2015---Friuli-Venezia-Giulia---Gestione-finanziaria-Spese-Enti-Locali.csv", 
                    encoding=result['encoding'],
                    delimiter=delimiter,
                    nrows=5)
    
    # Print the column names
    print("\nColumn names:")
    print(df.columns.tolist())
    
    # Print the first 5 rows
    print("\nFirst 5 rows:")
    print(df.head())
    
except Exception as e:
    print(f"Error reading CSV: {e}")
    
    # Try with different encoding/delimiter combinations
    print("\nTrying with different encoding/delimiter combinations...")
    for enc in ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']:
        for delim in [',', ';', '\t', '|']:
            try:
                print(f"Trying with encoding={enc}, delimiter='{delim}'")
                df = pd.read_csv("2015---Friuli-Venezia-Giulia---Gestione-finanziaria-Spese-Enti-Locali.csv", 
                                encoding=enc,
                                delimiter=delim,
                                nrows=5,
                                error_bad_lines=False)
                
                if len(df.columns) > 1:  # If we have more than one column, it's probably correct
                    print("Success!")
                    print("\nColumn names:")
                    print(df.columns.tolist())
                    
                    print("\nFirst 5 rows:")
                    print(df.head())
                    
                    # Save the working parameters
                    working_encoding = enc
                    working_delimiter = delim
                    break
            except Exception as e2:
                continue
        else:
            continue
        break
