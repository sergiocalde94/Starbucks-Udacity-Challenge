import pandas as pd
import pandas_profiling

from pathlib import Path

from .constants import PATH_DATA, PATH_REPORTS


def read_starbucks_data() -> pd.DataFrame:
    """Reads all starbucks data and formatted it
    getting a unique dataframe with all info
    """
    df_portfolio = pd.read_json(PATH_DATA / 'portfolio.json',
                                orient='records',
                                lines=True)

    df_profile = pd.read_json(PATH_DATA / 'profile.json',
                              orient='records',
                              lines=True)

    df_transcript = pd.read_json(PATH_DATA / 'transcript.json',
                                 orient='records',
                                 lines=True)

    df_portfolio['channels'] = (
        df_portfolio['channels']
        .apply(lambda x: '-'.join(map(str, x)))
    )

    columns_value = ['offer_id', 'amount', 'offer_id_aux', 'reward_expected']

    df_transcript[columns_value] = (
        pd.json_normalize(df_transcript['value'])
    )

    df_transcript['offer_id'] = (
        df_transcript['offer_id']
        .combine_first(df_transcript['offer_id_aux'])
    )

    df_transcript_joined_profile = (
        df_transcript
        .drop(columns=['value', 'offer_id_aux'])
        .merge(df_profile, left_on='person', right_on='id', how='left')
        .drop(columns='id')
    )

    df_transcript_joined_profile_and_portfolio = (
        df_transcript_joined_profile
        .merge(df_portfolio, left_on='offer_id', right_on='id', how='left')
        .drop(columns='id')
    )

    return df_transcript_joined_profile_and_portfolio


def pandas_profiling_to_file(df_to_profile: pd.DataFrame,
                             title: str,
                             output_filename: Path) -> None:
    """Alias for `pandas_profiling`, just takes `df_to_profile`
    and builds a report, saving it in `PATH_REPORTS` defined
    in submodule `constants.py`
    """
    if (PATH_REPORTS / output_filename).exists():
        print(f'File {output_filename} already exists')
    else:
        (df_to_profile
         .profile_report(title=title)
         .to_file(output_file=PATH_REPORTS / output_filename))
