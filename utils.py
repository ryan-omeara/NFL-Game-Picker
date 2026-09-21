#Managing and assembling all features
import pandas as pd
import nflreadpy as nfl

def assemble_features(schedules: pd.DataFrame) -> list:
    #Build Feature List
    team_features = [
        col for col in schedules.columns
        if col.startswith('home_team_') or col.startswith('away_team_')
    ]

    roof_features = [
        col for col in schedules.columns
        if col.startswith('roof_')
    ]

    surface_features = [
        col for col in schedules.columns
        if col.startswith('surface_')
    ]

    # Regular numerical features
    numeric_features = [
        'spread_line',
        'week',
        'year',
        'day_of_year',
        'gametime',
        'away_rest',
        'home_rest',
        'div_game',
        'home_moneyline',
        'away_moneyline',
    ]

    # Combine everything
    features = (
        numeric_features
        + roof_features
        + surface_features
        + team_features
    )
    return features

def build_schedules() -> pd.DataFrame: 
    schedules_polars = nfl.load_schedules(seasons=True)
    schedules = schedules_polars.to_pandas()
    schedules = schedules[schedules['game_type'] == 'REG'] #Filter for regular season games
    schedules = schedules.rename(columns={'season':'year'})

    #Day of Year (1-365)
    schedules['day_of_year'] = pd.to_datetime(schedules['gameday']).dt.dayofyear
    #Game Time (in minutes since midnight)
    time = pd.to_datetime(schedules['gametime'], format='%H:%M')
    #Rest Difference (home - away rest days)
    schedules['rest_diff'] = schedules['home_rest'] - schedules['away_rest']
    schedules['gametime'] =  (time.dt.hour * 60 + time.dt.minute)
    #Roof Type (Outdoors, Open, Closed, Dome)
    roofs = pd.get_dummies(schedules['roof'], prefix='roof', dtype=int)
    #Surfaces
    surfaces = pd.get_dummies(schedules['surface'], prefix='surface', dtype=int)
    # One-hot team data
    teams = pd.get_dummies(
        schedules[['home_team', 'away_team']],
        columns=['home_team', 'away_team'],
        dtype=int
    )
    schedules = pd.concat([schedules, teams, roofs, surfaces], axis=1)
    return schedules