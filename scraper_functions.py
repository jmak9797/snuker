
import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import numpy as np

def extract_score_and_break(score_str):
    """
    For a score format 
        '{total_frame_score}({highest_break})', but highest break only there if 50+. 
    e.g. '120(100)' or '80', 
    extract the total frame score and break value.
    
    If multiple breaks 50+, take highest.
    """
    break_match = re.findall(r'\((\d+(?:,\d+)*)\)', score_str)  # Find all numbers inside the parentheses
    score = re.sub(r'\(\d+(?:,\d+)*\)', '', score_str).strip()  # Remove parentheses and the break values

    if break_match:  # If breaks are found
        # Split the string by commas and convert each to an integer, then take the max
        break_values = [int(b) for b in break_match[0].split(',')]
        break_val = max(break_values)
    else:
        break_val = 0
    return int(score), break_val
    
def get_frame_level_result(match_class):
    """
    For a single match class (html format) strip out the details:
       - match-id, playuer names, best of val, match date, match score, frame scores.
    this gets a match level df.
    
    loops through each frame score and extract the total frame score and high break from each.
    Converts it to a frame level df.
     """
    df = match_class

    if "walkover" in str(df).lower():
        return pd.DataFrame()
    match_id = df['data-match-id']
    round_name = df.find('div', {'class': 'round_name'}).get_text(strip=True)
    player_1_name = df.find('div', {'class': 'player_1_name'}).get_text(strip=True)
    player_2_name = df.find('div', {'class': 'player_2_name'}).get_text(strip=True)
    best_of = df.find('span', {'class': 'best_of'}).get_text(strip=True)
    
    match_date = df.find('div', {'class': 'played_on'})
    if match_date is None:
        return pd.DataFrame()
    match_date=match_date.get_text(strip=True)
    
    player_1_score = df.find('span', {'class': 'player_1_score'}).get_text(strip=True)
    player_2_score = df.find('span', {'class': 'player_2_score'}).get_text(strip=True)
    
    frame_scores = df.find('div', {'class': 'frame_scores'})
    if frame_scores is None:
        return pd.DataFrame()
    frame_scores=frame_scores.get_text(strip=True)
    
    match_data = {
        "match_id": match_id,
        "round_name": round_name,
        "player_1_name": player_1_name,
        "player_2_name": player_2_name,
        "best_of": best_of,
        "match_date": match_date,
        "player_1_score": player_1_score,
        "player_2_score": player_2_score,
        "frame_scores": frame_scores
    }
    
    ## convert 
    # Split the frame_scores into individual frames
    frame_scores = match_data['frame_scores'].split(';')
    frame_numbers, player_1_scores, player_2_scores, player_1_breaks, player_2_breaks = [], [], [], [], [] 
    # Iterate through the frame scores and extract information
    #print(match_data)
    for frame_num, frame in enumerate(frame_scores, start=1):
        # Split the frame score into player 1 and player 2 scores
        try:
            frame_data = frame.split('-')
            player_1_frame_score, player_1_50_break = extract_score_and_break(frame_data[0].strip())
            if len(frame_data)>1: # strange thing Gary Wilson vs Noppon (2023-12-17) where one frame was 81(81). assume oppo is 0.
                player_2_frame_score, player_2_50_break = extract_score_and_break(frame_data[1].strip())
            else:
                player_2_frame_score, player_2_50_break = 0, 0
        except ValueError:
            player_1_frame_score, player_1_50_break = 0, 0
            player_2_frame_score, player_2_50_break = 0, 0
        
        frame_numbers.append(frame_num)
        player_1_scores.append(player_1_frame_score)
        player_2_scores.append(player_2_frame_score)
        player_1_breaks.append(player_1_50_break)
        player_2_breaks.append(player_2_50_break)
    df = pd.DataFrame({
        'match_id': int(match_data['match_id']),
        'round_name': match_data['round_name'],
        'player_1_name': match_data['player_1_name'],
        'player_2_name': match_data['player_2_name'],
        'best_of': match_data['best_of'],
        'match_date': match_data['match_date'],
        'player_1_score': match_data['player_1_score'],
        'player_2_score': match_data['player_2_score'],
        'frame_number': frame_numbers,
        'player_1_frame_score': player_1_scores,
        'player_2_frame_score': player_2_scores,
        'player_1_50_break': player_1_breaks,
        'player_2_50_break': player_2_breaks
    })
    df.best_of=df.best_of.str.replace("(", "").str.replace(")", "").astype(int)
    return df

def get_frame_level_data(soup):
    """
    The HTML for each match on cuetracker is labelled as classes = 'match row odd' or 'match row even'.
    Loops through each match on the page, gets frame level df for each and concat together to output.
    """
    
    odd_matches = soup.find_all('div', {'class': 'match row odd'})
    even_matches = soup.find_all('div', {'class': 'match row even'})
    match_list = []
    for match_odd in odd_matches:
        tmp = get_frame_level_result(match_odd)
        match_list.append(tmp)
    for match_even in even_matches:
        tmp = get_frame_level_result(match_even)
        match_list.append(tmp)
    return pd.concat(match_list)

def convert_matches_to_player_level(match_level, player_name):
    """
    Converts a dataframe being player1 and player2, where winning player is player1, to be player specific for player passed in.
    I.e. convert to player/oppo. 
    Allows for analysis on a player level easier.
    """
    df = match_level.copy()
    match_cols = [col for col in df if "player" not in col]
    
    p1 = df[df.player_1_name.str.replace("'", "").str.strip().str.lower() == player_name.replace("'", "").lower()].copy()
    p2 = df[df.player_2_name.str.replace("'", "").str.strip().str.lower() == player_name.replace("'", "").lower()].copy()
    
    p1.columns = p1.columns.str.replace("player_1", "player").str.replace("player_2", "oppo")
    p2.columns = p2.columns.str.replace("player_1", "oppo").str.replace("player_2", "player")
    _oppo_player_cols = [col for col in p2 if "oppo" in col]

    output = p1.merge(p2[["match_id", "frame_number"]+_oppo_player_cols], on=["match_id", "frame_number"], how='inner')
    return pd.concat([p1,p2]).reset_index(drop=True)

def load_frame_level_df_from_soup(soup_dict, player1_name, player2_name):
    """
    Wrapper to loop through each soup (player/season level) and get a match&frame level dataframe.
    Concat all seasons together to get a df for player1 and player2
    
    soup_dict: output of get_soups.
    
    """
    matches1_25 = get_frame_level_data(soup_dict["soup1_25"])
    matches1_24 = get_frame_level_data(soup_dict["soup1_24"])
    matches1_23 = get_frame_level_data(soup_dict["soup1_23"])
    matches1_22 = get_frame_level_data(soup_dict["soup1_22"])
    matches1_21 = get_frame_level_data(soup_dict["soup1_21"])
    matches1 = pd.concat([matches1_25,matches1_24,matches1_23,matches1_22,matches1_21])
    del matches1_25, matches1_24, matches1_23, matches1_22,matches1_21
    print(f"loaded player1 matches, count: {len(matches1)}")

    matches2_25 = get_frame_level_data(soup_dict["soup2_25"])
    matches2_24 = get_frame_level_data(soup_dict["soup2_24"])
    matches2_23 = get_frame_level_data(soup_dict["soup2_23"])
    matches2_22 = get_frame_level_data(soup_dict["soup2_22"])
    matches2_21 = get_frame_level_data(soup_dict["soup2_21"])
    matches2 = pd.concat([matches2_25,matches2_24,matches2_23,matches2_22,matches2_21])
    del matches2_25, matches2_24, matches2_23, matches2_22,matches2_21
    print(f"loaded player2 matches, count: {len(matches2)}")

    player1_df = convert_matches_to_player_level(matches1, player1_name.replace("-", " "))
    player2_df = convert_matches_to_player_level(matches2, player2_name.replace("-", " "))
    return player1_df, player2_df
