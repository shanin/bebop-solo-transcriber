import os
import urllib.request
import pandas as pd
import numpy as np
import sqlite3
import json
import argparse
import torch

SOLO_MISTAKES = {
    "FatsNavarro_GoodBait_No1_Solo": "FatsNavarro_GoodBait_Solo",
    "FatsNavarro_GoodBait_No2_Solo": "FatsNavarro_GoodBait_AlternateTake_Solo",
    "BranfordMarsalis_Ummg_Solo": "BranfordMarsalis_U.M.M.G._Solo",
    "SonnyRollins_I'llRememberApril-AlternateTake2_Solo": "SonnyRollins_I'llRememberApril_AlternateTake2_Solo",
    "PaulDesmond_BlueRondoAlaTurk_Solo": "PaulDesmond_BlueRondoALaTurk_Solo",
    "BranfordMarsalis_GutbucketSteepy_Solo": "BranfordMarsalis_GutBucketSteepy_Solo",
    "DizzyGillespie_Blue'NBoogie_Solo": "DizzyGillespie_Blue'nBoogie_Solo",
    "EricDolphy_Aisha_solo": "EricDolphy_Aisha_Solo",
    "KidOry_Who'sit_Solo": "KidOry_Who'sIt_Solo",
    "WayneShorter_JuJu_Solo": "WayneShorter_Juju_Solo",
}

def download_wjazzd_db(database_folder = 'artifacts', database_filename = 'wjazzd.db'):
    # Create artifacts directory if it doesn't exist
    os.makedirs(database_folder, exist_ok=True)

    # Download database file
    database_file_url = 'https://jazzomat.hfm-weimar.de/download/downloads/wjazzd.db'

    if not os.path.exists(os.path.join(database_folder, database_filename)):
        print(f"Downloading database from {database_file_url}...")
        urllib.request.urlretrieve(database_file_url, os.path.join(database_folder, database_filename))
        print("Download complete!")
    else:
        print("Database file already exists.")


def fetch_data(cursor, melid, table):
    cursor.execute(f"SELECT * FROM {table} WHERE melid = ?", (melid,))
    rows = cursor.fetchall()

    # Get column names
    cursor.execute(f"PRAGMA table_info({table})")
    columns = cursor.fetchall()
    column_names = [col[1] for col in columns]

    general_list = []
    for row in rows:
        row_dict = {}
        for col in column_names:
            row_dict[col] = row[column_names.index(col)]
        general_list.append(row_dict)

    return general_list



def prepare_sample(melid, cursor = None, database_folder = 'artifacts', database_filename = 'wjazzd.db'):
    if cursor is None:
        download_wjazzd_db(database_folder, database_filename)
        conn = sqlite3.connect(os.path.join(database_folder, database_filename))
        cursor = conn.cursor()

    melody_df = pd.DataFrame(fetch_data(cursor, melid, 'melody'))
    solo_info_df = pd.DataFrame(fetch_data(cursor, melid, 'solo_info'))
    beats_info_df = pd.DataFrame(fetch_data(cursor, melid, 'beats'))
    transcription_info_df = pd.DataFrame(fetch_data(cursor, melid, 'transcription_info'))

    min_bar_num = min(melody_df['bar'].min(), beats_info_df['bar'].min())
    melody_df['bar'] = melody_df['bar'] - min_bar_num
    beats_info_df['bar'] = beats_info_df['bar'] - min_bar_num

    audio_filename = transcription_info_df['filename_solo'].values[0]
    if audio_filename in SOLO_MISTAKES:
        audio_filename = SOLO_MISTAKES[audio_filename]
    audio_filename = f"{audio_filename}.wav"

    return {
        'id': f'WJD{str(melid).zfill(3)}',
        'audio_filename': audio_filename,
        'melody_df': melody_df,
        'solo_info_df': solo_info_df,
        'beats_info_df': beats_info_df,
        'transcription_info_df': transcription_info_df,
    }

