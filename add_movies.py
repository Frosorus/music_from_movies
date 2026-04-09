#!/usr/bin/env python3
"""
Incremental movie addition — skips movies already in the database.
Usage: python add_movies.py [db_path] [audio_subtitles.json]
"""

import sys
from pathlib import Path
from populate_database import create_database, run_pipeline

if __name__ == '__main__':
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'movie_clips.db'
    json_path = sys.argv[2] if len(sys.argv) > 2 else 'audio_subtitles.json'

    db = create_database(db_path)

    already_done = {r[0] for r in db.execute("SELECT DISTINCT movie_name FROM clips")}
    print(f"Already in DB: {len(already_done)} movies")

    total = run_pipeline(json_path, db, skip_movies=already_done)
    db.close()
    print(f"\nClips added: {total:,}")
