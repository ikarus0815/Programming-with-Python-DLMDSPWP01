import pandas as pd
import sqlite3
import os

from database_manager import DatabaseManager, DatabaseError


def test_create_and_load(tmp_path):
    dbpath = tmp_path / "test.db"
    dm = DatabaseManager(dbpath)
    dm.create_tables()
    # create small dataframes
    df_train = pd.DataFrame({'x':[0,1],'y1':[1,2],'y2':[3,4],'y3':[5,6],'y4':[7,8]})
    df_ideal = pd.DataFrame({'x':[0,1],'y1':[0,0],'y2':[1,1]})
    dm.load_training(df_train)
    dm.load_ideal(df_ideal)
    # open sqlite directly to check tables exist
    conn = sqlite3.connect(dbpath)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    names = {row[0] for row in cur.fetchall()}
    assert 'training' in names
    assert 'ideal' in names
    conn.close()


def test_store_test_results(tmp_path):
    dbpath = tmp_path / "test2.db"
    dm = DatabaseManager(dbpath)
    dm.create_tables()
    df = pd.DataFrame({'x':[0],'y_test':[0],'delta_y':[0],'ideal_func':[1]})
    dm.store_test_results(df)
    conn = sqlite3.connect(dbpath)
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM test_mapping")
    count = cur.fetchone()[0]
    assert count == 1
    conn.close()
