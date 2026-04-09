
import time
import pandas as pd
import numpy as np
import wbgapi as wb
import country_converter as coco


def missing_report(dataframe):
    """Return a sorted DataFrame showing missing count and % per column."""
    missing = dataframe.isnull().sum()
    pct = (missing / len(dataframe) * 100).round(2)
    report = pd.DataFrame({'missing_count': missing, 'missing_pct': pct})
    return report[report.missing_count > 0].sort_values('missing_pct', ascending=False)



"""

World Bank Data Fetcher

Run this script separately to fetch WDI and WGI indicators from the World Bank API.
It produces two CSV files that the main wrangling notebook will import:
  - wdi_raw.csv   (World Development Indicators)
  - wgi_raw.csv   (Worldwide Governance Indicators)

The script expects 'World Happiness Report.csv' to be in the same directory, 
so it won't run until you move it there.

Fetching can take 10-30 minutes depending on API server load.

"""



def fetch_wb(iso3_list, years, indicators, db_type=2, batch_size=None, country_batch_size=50, max_retries=5, backoff=10):
    """Fetch World Bank data, batching both indicators and countries to keep URLs short."""

    db_name = {2: 'WDI', 3: 'WGI'}.get(db_type, f'db={db_type}')
    ind_batches = (
        [indicators[i:i+batch_size] for i in range(0, len(indicators), batch_size)]
        if batch_size else [indicators]
    )
    country_batches = [iso3_list[i:i+country_batch_size] for i in range(0, len(iso3_list), country_batch_size)]

    total = len(ind_batches) * len(country_batches)
    results = []
    failed = 0
    n = 0

    for ind_batch in ind_batches:
        for country_batch in country_batches:
            n += 1
            for attempt in range(1, max_retries + 1):
                try:
                    df = wb.data.DataFrame(
                        ind_batch, economy=country_batch,
                        time=years, db=db_type, labels=False
                    ).reset_index()
                    df.rename(columns={'economy': 'ISO3', 'time': 'Year'}, inplace=True)
                    results.append(df)
                    print(f'  [{db_name}] Batch {n}/{total} fetched successfully.')
                    time.sleep(1)
                    break
                except Exception as e:
                    wait = backoff * (2 ** (attempt - 1))
                    if attempt < max_retries:
                        print(f'  [{db_name}] Batch {n}/{total} attempt {attempt} failed, retrying in {wait}s...')
                        time.sleep(wait)
                    else:
                        print(f'  [{db_name}] Batch {n}/{total} all {max_retries} attempts failed. ({type(e).__name__}: {e})')
                        failed += 1

    if not results:
        print(f'[{db_name}] All {total} batches failed.')
        return pd.DataFrame(columns=['ISO3', 'series'])

    if failed:
        print(f'[{db_name}] Done — {len(results)}/{total} batches succeeded ({failed} failed).')
    else:
        print(f'[{db_name}] Done — all {total} batches succeeded.')
    return pd.concat(results, ignore_index=True)


if __name__ == '__main__':

    # ── Load happiness data to get country list and year range ──
    happiness_df = pd.read_csv('drafts/datasets/World Happiness Report.csv')

    cc = coco.CountryConverter()
    unique_countries = happiness_df['Country Name'].unique()
    iso3_series = cc.pandas_convert(pd.Series(unique_countries), to='ISO3')
    iso3_series = iso3_series.replace('not found', np.nan)
    iso_map = dict(zip(unique_countries, iso3_series))
    happiness_df['ISO3'] = happiness_df['Country Name'].map(iso_map)

    years = sorted(happiness_df['Year'].unique())
    iso3_list = happiness_df['ISO3'].dropna().unique().tolist()

    # ── Define indicators ──
    WB_INDICATORS = {
        'NY.GDP.PCAP.CD':      'GDP/Capita',
        'NY.GDP.MKTP.KD.ZG':   'GDP Growth Rate',
        'FP.CPI.TOTL.ZG':      'Inflation Rate',
        'NE.TRD.GNFS.ZS':      'Trade Openness',
        'SP.DYN.LE00.IN':      'Life Expectancy',
        'SH.DYN.MORT':         'Child Mortality Rate',
        'SH.XPD.CHEX.GD.ZS':  'Health Expenditure pct of GDP',
        'SH.XPD.OOPC.CH.ZS':  'Out Of Pocket Health pct',
        'SE.COM.DURS':         'Expected Years Of Schooling',
        'IT.NET.USER.ZS':      'Internet Penetration',
        'EG.ELC.ACCS.ZS':     'Electricity Access pct',
        'SP.URB.TOTL.IN.ZS':  'Urbanization Rate',
        'IS.ROD.PAVE.ZS':     'Paved Roads pct',
        'SL.UEM.TOTL.ZS':     'Unemployment Rate',
        'SL.TLF.CACT.FE.ZS':  'Female Labor Participation',
        'SP.POP.TOTL':         'Population',
        'SP.POP.GROW':         'Population Growth Rate',
    }

    WGI_INDICATORS = {
        'CC.EST':  'Control Of Corruption',
        'GE.EST':  'Govt Effectiveness',
        'RL.EST':  'Rule Of Law',
        'VA.EST':  'Voice Accountability',
        'PV.EST':  'Political Stability',
        'RQ.EST':  'Regulatory Quality',
    }

    # ── Fetch ──
    print(f'Fetching data for {len(iso3_list)} countries, years {years[0]}–{years[-1]}')
    print()

    print('Fetching WDI indicators...')
    df_wdi = fetch_wb(iso3_list, years, list(WB_INDICATORS.keys()), db_type=2, batch_size=6)

    print()
    print('Fetching WGI indicators...')
    df_wgi = fetch_wb(iso3_list, years, list(WGI_INDICATORS.keys()), db_type=3)

    # ── Save raw outputs (no transformations) ──
    df_wdi.to_csv('wdi_raw.csv', index=False)
    df_wgi.to_csv('wgi_raw.csv', index=False)

    print()
    print(f'Saved wdi_raw.csv  ({df_wdi.shape[0]:,} rows * {df_wdi.shape[1]} cols)')
    print(f'Saved wgi_raw.csv  ({df_wgi.shape[0]:,} rows * {df_wgi.shape[1]} cols)')
    print('Done. These files are ready for the wrangling notebook.')
