#!/usr/bin/env python3
"""
ensdf_search.py - Standalone Nuclear Isotope Search Engine & ENSDF Database Builder
Part of python-cmat (https://github.com/rlica/python-cmat)

Features:
- One-time fast parser and SQLite indexer for raw ENSDF files (ensdf.001 - ensdf.295)
- Sub-millisecond 1D photopeak identification with mass (A) and half-life (T1/2) filtering
- 2D coincidence cascade search with level topology matching and confidence scoring
- Direct ingestion and automated identification of fit_results_*.txt logs
- Standalone CLI interface and reusable Python API for webviewers
"""

import os
import sys
import glob
import re
import math
import time
import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any

# Periodic Table mapping element symbol -> Z and Z -> symbol
ELEMENT_Z_MAP = {
    "H": 1, "HE": 2, "LI": 3, "BE": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "NE": 10,
    "NA": 11, "MG": 12, "AL": 13, "SI": 14, "P": 15, "S": 16, "CL": 17, "AR": 18, "K": 19, "CA": 20,
    "SC": 21, "TI": 22, "V": 23, "CR": 24, "MN": 25, "FE": 26, "CO": 27, "NI": 28, "CU": 29, "ZN": 30,
    "GA": 31, "GE": 32, "AS": 33, "SE": 34, "BR": 35, "KR": 36, "RB": 37, "SR": 38, "Y": 39, "ZR": 40,
    "NB": 41, "MO": 42, "TC": 43, "RU": 44, "RH": 45, "PD": 46, "AG": 47, "CD": 48, "IN": 49, "SN": 50,
    "SB": 51, "TE": 52, "I": 53, "XE": 54, "CS": 55, "BA": 56, "LA": 57, "CE": 58, "PR": 59, "ND": 60,
    "PM": 61, "SM": 62, "EU": 63, "GD": 64, "TB": 65, "DY": 66, "HO": 67, "ER": 68, "TM": 69, "YB": 70,
    "LU": 71, "HF": 72, "TA": 73, "W": 74, "RE": 75, "OS": 76, "IR": 77, "PT": 78, "AU": 79, "HG": 80,
    "TL": 81, "PB": 82, "BI": 83, "PO": 84, "AT": 85, "RN": 86, "FR": 87, "RA": 88, "AC": 89, "TH": 90,
    "PA": 91, "U": 92, "NP": 93, "PU": 94, "AM": 95, "CM": 96, "BK": 97, "CF": 98, "ES": 99, "FM": 100,
    "MD": 101, "NO": 102, "LR": 103, "RF": 104, "DB": 105, "SG": 106, "BH": 107, "HS": 108, "MT": 109,
    "DS": 110, "RG": 111, "CN": 112, "NH": 113, "FL": 114, "MC": 115, "LV": 116, "TS": 117, "OG": 118,
    "NN": 0
}
Z_ELEMENT_MAP = {v: k for k, v in ELEMENT_Z_MAP.items() if k != "NN"}

DEFAULT_DB_FILENAME = "ensdf.db"
DEFAULT_ENSDF_DIR = "/Users/razvanlica/Downloads/ensdf_260901"


def parse_halflife(s: str) -> Tuple[Optional[float], str]:
    """
    Parse an ENSDF half-life string into (seconds, original_clean_string).
    Handles STABLE, standard units (Y, D, H, M, S, MS, US, NS, PS, FS, AS)
    and energy widths (EV, KEV, MEV via Gamma = hbar / tau).
    """
    if not s:
        return None, ""
    s_clean = s.strip()
    if not s_clean:
        return None, ""
    s_upper = s_clean.upper()

    if "STABLE" in s_upper:
        return 1.0e30, "STABLE"
    if "?" in s_clean:
        s_clean_noq = s_clean.replace("?", "").strip()
    else:
        s_clean_noq = s_clean

    m = re.match(r"^([0-9\.]+)\s*(SEC|MIN|KEV|MEV|YR|HR|MS|US|NS|PS|FS|AS|EV|Y|D|H|M|S)\b", s_clean_noq, re.IGNORECASE)
    if not m:
        return None, s_clean

    try:
        val = float(m.group(1))
    except ValueError:
        return None, s_clean

    unit = m.group(2).upper()
    factors = {
        "Y": 365.25 * 86400.0,
        "YR": 365.25 * 86400.0,
        "D": 86400.0,
        "H": 3600.0,
        "HR": 3600.0,
        "M": 60.0,
        "MIN": 60.0,
        "S": 1.0,
        "SEC": 1.0,
        "MS": 1e-3,
        "US": 1e-6,
        "NS": 1e-9,
        "PS": 1e-12,
        "FS": 1e-15,
        "AS": 1e-18,
    }

    if unit in factors:
        return val * factors[unit], s_clean
    elif unit in ("EV", "KEV", "MEV"):
        ev_val = val * (1e6 if unit == "MEV" else (1e3 if unit == "KEV" else 1.0))
        if ev_val > 0:
            return (6.582119e-16 * 0.69314718) / ev_val, s_clean

    return None, s_clean


def parse_human_duration(s: str) -> Optional[float]:
    """Parse user-friendly duration strings like '1s', '10m', '2.5h', '5d', '1y' into seconds."""
    if not s or str(s).strip() == "":
        return None
    s = str(s).strip().lower()
    if s in ("stable", "inf", "infinity"):
        return 1e29
    m = re.match(r"^([0-9\.]+)\s*([a-z]+)?$", s)
    if not m:
        return None
    val = float(m.group(1))
    u = m.group(2) or "s"
    if u in ("s", "sec", "second", "seconds"):
        return val
    elif u in ("m", "min", "minute", "minutes"):
        return val * 60.0
    elif u in ("h", "hr", "hour", "hours"):
        return val * 3600.0
    elif u in ("d", "day", "days"):
        return val * 86400.0
    elif u in ("y", "yr", "year", "years"):
        return val * 365.25 * 86400.0
    elif u in ("ms", "msec"):
        return val * 1e-3
    elif u in ("us", "usec", "µs"):
        return val * 1e-6
    elif u in ("ns", "nsec"):
        return val * 1e-9
    return val


def parse_float_safe(s: str) -> Optional[float]:
    """Extract first floating-point number from string."""
    if not s:
        return None
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", s.strip())
    if m:
        try:
            return float(m.group(0))
        except Exception:
            return None
    return None


def parse_nuid(nuid_str: str) -> Tuple[Optional[int], Optional[int], str, str]:
    """
    Parse an ENSDF NUID (e.g. ' 60NI', ' 60CO', '  1H ', '152EU')
    into (A, Z, element, clean_nuid).
    """
    s = nuid_str.strip().upper()
    m = re.match(r"^(\d+)\s*([A-Z]+)$", s)
    if not m:
        return None, None, s, s
    a = int(m.group(1))
    elem = m.group(2)
    z = ELEMENT_Z_MAP.get(elem)
    clean = f"{a}{elem.capitalize()}"
    return a, z, elem.capitalize(), clean


# ==============================================================================
# Database Creation & Indexing Engine
# ==============================================================================

def init_ensdf_database(conn: sqlite3.Connection):
    """Create optimized tables and indexes for ENSDF data."""
    cur = conn.cursor()
    cur.execute("PRAGMA journal_mode = WAL;")
    cur.execute("PRAGMA synchronous = NORMAL;")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS nuclides (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nuid TEXT UNIQUE,
        z INTEGER,
        a INTEGER,
        element TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS datasets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nuclide_id INTEGER REFERENCES nuclides(id),
        dsid TEXT,
        ds_type TEXT,
        parent_nuclide TEXT,
        parent_z INTEGER,
        parent_a INTEGER,
        parent_energy REAL,
        parent_jpi TEXT,
        parent_t12_s REAL,
        parent_t12_str TEXT,
        q_val REAL
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS levels (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER REFERENCES datasets(id),
        nuclide_id INTEGER REFERENCES nuclides(id),
        energy REAL,
        energy_err REAL,
        jpi TEXT,
        t12_s REAL,
        t12_str TEXT
    );
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS gammas (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        dataset_id INTEGER REFERENCES datasets(id),
        nuclide_id INTEGER REFERENCES nuclides(id),
        level_id INTEGER REFERENCES levels(id),
        energy REAL,
        energy_err REAL,
        intensity REAL,
        intensity_err REAL,
        multipolarity TEXT,
        final_level_energy REAL,
        conv_coeff REAL
    );
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_gammas_energy ON gammas(energy);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_gammas_ds ON gammas(dataset_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_gammas_nuclide ON gammas(nuclide_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_levels_ds ON levels(dataset_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_levels_energy ON levels(energy);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_datasets_nuclide ON datasets(nuclide_id);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_datasets_parent_t12 ON datasets(parent_t12_s);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_nuclides_a ON nuclides(a);")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_nuclides_z ON nuclides(z);")

    conn.commit()


def build_ensdf_database(ensdf_dir: str, db_path: str = DEFAULT_DB_FILENAME, verbose: bool = True) -> int:
    """
    Parse all ensdf.* files in ensdf_dir and populate the indexed SQLite database.
    Processes all 300 mass files in batches for maximum insertion speed.
    """
    ensdf_path = Path(ensdf_dir)
    if not ensdf_path.exists() or not ensdf_path.is_dir():
        raise FileNotFoundError(f"ENSDF directory not found: {ensdf_dir}")

    files = sorted(glob.glob(str(ensdf_path / "ensdf.*")))
    if not files:
        raise FileNotFoundError(f"No ensdf.* files found in {ensdf_dir}")

    db_file = Path(db_path).resolve()
    if db_file.exists():
        try:
            db_file.unlink()
        except Exception:
            pass

    conn = sqlite3.connect(str(db_file))
    init_ensdf_database(conn)
    cur = conn.cursor()

    t0 = time.time()
    if verbose:
        print(f"[*] Building local ENSDF database at: {db_file}")
        print(f"[*] Ingesting {len(files)} ENSDF data files from {ensdf_path.name}/ ...")

    nuclide_cache: Dict[str, int] = {}
    total_gammas = 0
    total_levels = 0
    total_datasets = 0

    batch_datasets = []
    batch_levels = []
    batch_gammas = []

    def get_or_create_nuclide_id(nuid_str: str) -> Optional[int]:
        s = nuid_str.strip().upper()
        if not s:
            return None
        if s in nuclide_cache:
            return nuclide_cache[s]
        a, z, elem, clean = parse_nuid(s)
        cur.execute("INSERT OR IGNORE INTO nuclides (nuid, z, a, element) VALUES (?, ?, ?, ?)", (clean, z, a, elem))
        cur.execute("SELECT id FROM nuclides WHERE nuid = ?", (clean,))
        row = cur.fetchone()
        if row:
            nuclide_cache[s] = row[0]
            nuclide_cache[clean] = row[0]
            return row[0]
        return None

    dataset_counter = 0
    level_counter = 0
    gamma_counter = 0

    for idx, fpath in enumerate(files):
        with open(fpath, "r", encoding="latin-1") as f:
            current_dataset = None
            current_ds_id = None
            current_nuclide_id = None
            current_level_id = None
            current_level_energy = 0.0

            levels_in_dataset = []

            for line in f:
                if len(line.strip()) == 0:
                    current_dataset = None
                    current_ds_id = None
                    current_level_id = None
                    levels_in_dataset = []
                    continue

                if len(line) < 8:
                    continue

                col_tag = line[5:8]

                # 1. Dataset Identification Record
                if current_ds_id is None:
                    nuid_raw = line[0:5].strip()
                    dsid_raw = line[9:39].strip()
                    if dsid_raw:
                        dataset_counter += 1
                        current_ds_id = dataset_counter
                        current_nuclide_id = get_or_create_nuclide_id(nuid_raw)

                        ds_upper = dsid_raw.upper()
                        if "ADOPTED" in ds_upper:
                            ds_type = "adopted"
                        elif "DECAY" in ds_upper:
                            ds_type = "decay"
                        else:
                            ds_type = "reaction"

                        current_dataset = {
                            "id": current_ds_id,
                            "nuclide_id": current_nuclide_id,
                            "dsid": dsid_raw,
                            "ds_type": ds_type,
                            "parent_nuclide": None,
                            "parent_z": None,
                            "parent_a": None,
                            "parent_energy": None,
                            "parent_jpi": None,
                            "parent_t12_s": None,
                            "parent_t12_str": None,
                            "q_val": None,
                        }
                        batch_datasets.append(current_dataset)
                        total_datasets += 1
                        continue

                # 2. Parent Record (P)
                if col_tag == "  P" and current_dataset is not None:
                    p_nuid = line[0:5].strip()
                    p_a, p_z, _, p_clean = parse_nuid(p_nuid)
                    p_e = parse_float_safe(line[9:19])
                    p_jpi = line[21:39].strip() or None
                    p_t12_s, p_t12_str = parse_halflife(line[39:49])
                    q_val = parse_float_safe(line[64:74])

                    current_dataset["parent_nuclide"] = p_clean or p_nuid
                    current_dataset["parent_a"] = p_a
                    current_dataset["parent_z"] = p_z
                    current_dataset["parent_energy"] = p_e
                    current_dataset["parent_jpi"] = p_jpi
                    current_dataset["parent_t12_s"] = p_t12_s
                    current_dataset["parent_t12_str"] = p_t12_str
                    current_dataset["q_val"] = q_val
                    continue

                # 3. Level Record (L)
                if col_tag == "  L" and current_ds_id is not None:
                    lev_e = parse_float_safe(line[9:19])
                    if lev_e is not None:
                        level_counter += 1
                        current_level_id = level_counter
                        current_level_energy = lev_e

                        de_str = line[19:21].strip()
                        lev_de = parse_float_safe(de_str)
                        lev_jpi = line[21:39].strip() or None
                        lev_t12_s, lev_t12_str = parse_halflife(line[39:49])

                        batch_levels.append((
                            current_level_id,
                            current_ds_id,
                            current_nuclide_id,
                            lev_e,
                            lev_de,
                            lev_jpi,
                            lev_t12_s,
                            lev_t12_str
                        ))
                        levels_in_dataset.append((current_level_id, lev_e))
                        total_levels += 1
                    else:
                        current_level_id = None
                    continue

                # 4. Gamma Record (G)
                if col_tag == "  G" and current_ds_id is not None and current_level_id is not None:
                    g_e = parse_float_safe(line[9:19])
                    if g_e is not None and g_e > 0.1:
                        gamma_counter += 1
                        g_de = parse_float_safe(line[19:21])
                        g_ri = parse_float_safe(line[21:29])
                        g_dri = parse_float_safe(line[29:31])
                        g_m = line[31:41].strip() or None
                        g_cc = parse_float_safe(line[55:62])

                        calc_ef = current_level_energy - g_e
                        best_ef = None
                        min_diff = 999.0
                        for _, ef_candidate in levels_in_dataset:
                            diff = abs(ef_candidate - calc_ef)
                            if diff < min_diff and diff < 3.0:
                                min_diff = diff
                                best_ef = ef_candidate

                        batch_gammas.append((
                            gamma_counter,
                            current_ds_id,
                            current_nuclide_id,
                            current_level_id,
                            g_e,
                            g_de,
                            g_ri,
                            g_dri,
                            g_m,
                            best_ef if best_ef is not None else max(0.0, calc_ef),
                            g_cc
                        ))
                        total_gammas += 1
                    continue

        if len(batch_gammas) >= 50000:
            _flush_batches(cur, conn, batch_datasets, batch_levels, batch_gammas)
            batch_datasets = []
            batch_levels = []
            batch_gammas = []
            if verbose and (idx + 1) % 50 == 0:
                print(f"    - Processed {idx + 1}/{len(files)} files ({total_gammas:,} gammas indexed)...")

    _flush_batches(cur, conn, batch_datasets, batch_levels, batch_gammas)
    conn.commit()

    cur.execute("""
    CREATE VIEW IF NOT EXISTS v_gammas AS
    SELECT g.id AS gamma_id, g.energy AS gamma_energy, g.energy_err AS gamma_err,
           g.intensity, g.multipolarity, g.final_level_energy,
           l.id AS initial_level_id, l.energy AS initial_level_energy, l.jpi AS initial_jpi, l.t12_str AS initial_t12,
           d.id AS dataset_id, d.dsid, d.ds_type, d.parent_nuclide, d.parent_t12_s, d.parent_t12_str,
           n.id AS nuclide_id, n.nuid, n.z, n.a, n.element
    FROM gammas g
    JOIN levels l ON g.level_id = l.id
    JOIN datasets d ON g.dataset_id = d.id
    JOIN nuclides n ON g.nuclide_id = n.id;
    """)
    conn.commit()
    conn.close()

    elapsed = time.time() - t0
    db_size_mb = Path(db_path).stat().st_size / (1024 * 1024)
    if verbose:
        print(f"[+] ENSDF Database Build Complete in {elapsed:.1f}s!")
        print(f"    - Indexed: {len(nuclide_cache):,} nuclides, {total_datasets:,} datasets, {total_levels:,} levels, {total_gammas:,} transitions")
        print(f"    - Database Size: {db_size_mb:.1f} MB ({db_path})\n")

    return total_gammas


def _flush_batches(cur: sqlite3.Cursor, conn: sqlite3.Connection, datasets, levels, gammas):
    """Batch-insert datasets, levels, and gammas into SQLite."""
    if datasets:
        cur.executemany("""
        INSERT OR REPLACE INTO datasets (id, nuclide_id, dsid, ds_type, parent_nuclide, parent_z, parent_a,
                                        parent_energy, parent_jpi, parent_t12_s, parent_t12_str, q_val)
        VALUES (:id, :nuclide_id, :dsid, :ds_type, :parent_nuclide, :parent_z, :parent_a,
                :parent_energy, :parent_jpi, :parent_t12_s, :parent_t12_str, :q_val)
        """, datasets)

    if levels:
        cur.executemany("""
        INSERT OR REPLACE INTO levels (id, dataset_id, nuclide_id, energy, energy_err, jpi, t12_s, t12_str)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, levels)

    if gammas:
        cur.executemany("""
        INSERT OR REPLACE INTO gammas (id, dataset_id, nuclide_id, level_id, energy, energy_err,
                                       intensity, intensity_err, multipolarity, final_level_energy, conv_coeff)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, gammas)

    conn.commit()


# ==============================================================================
# Isotope Search & Coincidence Identification Engine
# ==============================================================================

class ENSDFSearchEngine:
    """Fast local search engine for 1D photopeaks and 2D coincidence cascade pairs."""

    def __init__(self, db_path: str = DEFAULT_DB_FILENAME):
        self.db_path = Path(db_path).resolve()
        if not self.db_path.exists():
            cwd_db = Path.cwd() / DEFAULT_DB_FILENAME
            if cwd_db.exists():
                self.db_path = cwd_db
            else:
                script_db = Path(__file__).resolve().parent / DEFAULT_DB_FILENAME
                if script_db.exists():
                    self.db_path = script_db

        self._ensure_db_decompressed()
        self._conn = None

    def _ensure_db_decompressed(self):
        """If ensdf.db does not exist, check for ensdf.db.gz and decompress automatically in ~0.2s."""
        if not self.db_path.exists() or self.db_path.stat().st_size < 1024 * 1024:
            gz_candidates = [
                self.db_path.with_name(self.db_path.name + ".gz"),
                self.db_path.parent / (DEFAULT_DB_FILENAME + ".gz"),
                Path.cwd() / (DEFAULT_DB_FILENAME + ".gz"),
                Path(__file__).resolve().parent / (DEFAULT_DB_FILENAME + ".gz"),
            ]
            for gz in gz_candidates:
                if gz.exists() and gz.stat().st_size > 1024:
                    import gzip, shutil
                    target_db = self.db_path if self.db_path.name.endswith(".db") else (Path(__file__).resolve().parent / DEFAULT_DB_FILENAME)
                    tmp_target = target_db.with_suffix(".tmp")
                    try:
                        print(f"📦 Unpacking local ENSDF database from {gz.name}...", file=sys.stderr)
                        with gzip.open(gz, "rb") as f_in:
                            with open(tmp_target, "wb") as f_out:
                                shutil.copyfileobj(f_in, f_out)
                        tmp_target.replace(target_db)
                        self.db_path = target_db
                        print(f"✅ ENSDF database ready ({target_db.name}, {target_db.stat().st_size / (1024*1024):.1f} MB)", file=sys.stderr)
                    except Exception as e:
                        if tmp_target.exists():
                            try: tmp_target.unlink()
                            except Exception: pass
                        print(f"⚠️ Warning: Auto-unpack of {gz.name} failed: {e}", file=sys.stderr)
                    break

    def is_available(self) -> bool:
        """Check if local database is built and ready."""
        self._ensure_db_decompressed()
        return self.db_path.exists() and self.db_path.stat().st_size > 1024 * 1024

    def get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            if not self.is_available():
                raise FileNotFoundError(
                    f"ENSDF database not found at {self.db_path}. Build it first with: python3 ensdf_search.py --build-db"
                )
            self._conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def get_stats(self) -> Dict[str, Any]:
        """Return database summary statistics."""
        if not self.is_available():
            return {"available": False, "db_path": str(self.db_path)}
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM nuclides")
        n_nuclides = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM datasets")
        n_datasets = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM levels")
        n_levels = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM gammas")
        n_gammas = cur.fetchone()[0]
        size_mb = round(self.db_path.stat().st_size / (1024 * 1024), 1)

        return {
            "available": True,
            "db_path": str(self.db_path),
            "size_mb": size_mb,
            "nuclides": n_nuclides,
            "datasets": n_datasets,
            "levels": n_levels,
            "gammas": n_gammas,
        }

    def search_1d(
        self,
        energy: float,
        tol: float = 1.5,
        a_min: Optional[int] = None,
        a_max: Optional[int] = None,
        elements: Optional[List[str]] = None,
        min_t12_s: Optional[float] = None,
        max_t12_s: Optional[float] = None,
        dataset_type: Optional[str] = None,
        limit: int = 40,
    ) -> List[Dict[str, Any]]:
        """
        Search for candidate transitions matching single energy E ± tol with nuclear constraints.
        Returns candidate list ranked by composite score.
        """
        conn = self.get_connection()
        cur = conn.cursor()

        query = """
        SELECT g.id AS gamma_id, g.energy, g.energy_err, g.intensity, g.multipolarity, g.final_level_energy,
               l.energy AS initial_level_energy, l.jpi AS initial_jpi, l.t12_str AS level_t12,
               d.id AS dataset_id, d.dsid, d.ds_type, d.parent_nuclide, d.parent_t12_s, d.parent_t12_str,
               n.nuid, n.z, n.a, n.element
        FROM gammas g
        JOIN levels l ON g.level_id = l.id
        JOIN datasets d ON g.dataset_id = d.id
        JOIN nuclides n ON g.nuclide_id = n.id
        WHERE g.energy BETWEEN ? AND ?
        """
        params = [energy - tol, energy + tol]

        if a_min is not None:
            query += " AND n.a >= ?"
            params.append(a_min)
        if a_max is not None:
            query += " AND n.a <= ?"
            params.append(a_max)

        if elements:
            elems_clean = [e.strip().capitalize() for e in elements if e.strip()]
            if elems_clean:
                placeholders = ",".join("?" for _ in elems_clean)
                query += f" AND n.element IN ({placeholders})"
                params.extend(elems_clean)

        if min_t12_s is not None:
            query += " AND d.parent_t12_s >= ?"
            params.append(min_t12_s)
        if max_t12_s is not None:
            query += " AND d.parent_t12_s <= ?"
            params.append(max_t12_s)

        if dataset_type and dataset_type != "all":
            query += " AND d.ds_type = ?"
            params.append(dataset_type)

        query += " ORDER BY ABS(g.energy - ?) ASC LIMIT ?"
        params.extend([energy, limit])

        cur.execute(query, params)
        rows = cur.fetchall()

        results = []
        for r in rows:
            g_e = float(r["energy"])
            diff = abs(g_e - energy)
            rel_int = float(r["intensity"]) if r["intensity"] is not None else 1.0

            sigma = max(0.2, tol / 2.0)
            score = math.exp(-0.5 * (diff / sigma) ** 2) * min(100.0, max(1.0, rel_int))

            results.append({
                "nuclide": r["nuid"],
                "z": r["z"],
                "a": r["a"],
                "element": r["element"],
                "gamma_energy": g_e,
                "gamma_err": r["energy_err"],
                "energy_diff": round(g_e - energy, 3),
                "intensity": r["intensity"],
                "multipolarity": r["multipolarity"],
                "initial_level": r["initial_level_energy"],
                "final_level": r["final_level_energy"],
                "initial_jpi": r["initial_jpi"],
                "dataset": r["dsid"],
                "ds_type": r["ds_type"],
                "parent_nuclide": r["parent_nuclide"] or r["nuid"],
                "parent_halflife": r["parent_t12_str"] or r["level_t12"] or "Prompt / In-Beam",
                "score": round(score, 1),
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    def search_2d(
        self,
        energy1: float,
        energy2: float,
        tol1: float = 2.0,
        tol2: float = 2.0,
        a_min: Optional[int] = None,
        a_max: Optional[int] = None,
        elements: Optional[List[str]] = None,
        min_t12_s: Optional[float] = None,
        max_t12_s: Optional[float] = None,
        dataset_type: Optional[str] = None,
        limit: int = 5,
        unique_nuclides: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Search for 2D coincidence pair (E1, E2) within the same nuclide dataset.
        Prioritizes direct decay cascades (L_a -> L_b -> L_c), then sequential cascades,
        while damping coincidences originating from large level gaps or very high excitation energies.
        Returns up to `limit` top likely candidate isotopes.
        """
        conn = self.get_connection()
        cur = conn.cursor()

        query1 = "SELECT id, dataset_id, nuclide_id, energy, intensity, level_id, final_level_energy FROM gammas WHERE energy BETWEEN ? AND ?"
        cur.execute(query1, (energy1 - tol1, energy1 + tol1))
        hits1 = cur.fetchall()

        cur.execute(query1, (energy2 - tol2, energy2 + tol2))
        hits2 = cur.fetchall()

        ds_hits1: Dict[int, list] = {}
        for h in hits1:
            ds_hits1.setdefault(h["dataset_id"], []).append(h)

        ds_hits2: Dict[int, list] = {}
        for h in hits2:
            ds_hits2.setdefault(h["dataset_id"], []).append(h)

        common_ds_ids = set(ds_hits1.keys()).intersection(set(ds_hits2.keys()))
        if not common_ds_ids:
            return []

        placeholders = ",".join("?" for _ in common_ds_ids)
        ds_query = f"""
        SELECT d.id AS ds_id, d.dsid, d.ds_type, d.parent_nuclide, d.parent_t12_s, d.parent_t12_str,
               n.nuid, n.z, n.a, n.element
        FROM datasets d
        JOIN nuclides n ON d.nuclide_id = n.id
        WHERE d.id IN ({placeholders})
        """
        ds_params = list(common_ds_ids)

        if a_min is not None:
            ds_query += " AND n.a >= ?"
            ds_params.append(a_min)
        if a_max is not None:
            ds_query += " AND n.a <= ?"
            ds_params.append(a_max)

        if elements:
            elems_clean = [e.strip().capitalize() for e in elements if e.strip()]
            if elems_clean:
                pl_el = ",".join("?" for _ in elems_clean)
                ds_query += f" AND n.element IN ({pl_el})"
                ds_params.extend(elems_clean)

        if min_t12_s is not None:
            ds_query += " AND d.parent_t12_s >= ?"
            ds_params.append(min_t12_s)
        if max_t12_s is not None:
            ds_query += " AND d.parent_t12_s <= ?"
            ds_params.append(max_t12_s)

        if dataset_type and dataset_type != "all":
            ds_query += " AND d.ds_type = ?"
            ds_params.append(dataset_type)

        cur.execute(ds_query, ds_params)
        valid_datasets = {r["ds_id"]: r for r in cur.fetchall()}

        candidates = []
        for ds_id, ds_meta in valid_datasets.items():
            for g1 in ds_hits1[ds_id]:
                for g2 in ds_hits2[ds_id]:
                    if g1["id"] == g2["id"]:
                        continue

                    e1_val = float(g1["energy"])
                    e2_val = float(g2["energy"])

                    diff1 = abs(e1_val - energy1)
                    diff2 = abs(e2_val - energy2)

                    cur.execute("SELECT energy, jpi FROM levels WHERE id = ?", (g1["level_id"],))
                    l1_row = cur.fetchone()
                    l1_e = l1_row["energy"] if l1_row else 0.0
                    l1_jpi = l1_row["jpi"] if l1_row else ""

                    cur.execute("SELECT energy, jpi FROM levels WHERE id = ?", (g2["level_id"],))
                    l2_row = cur.fetchone()
                    l2_e = l2_row["energy"] if l2_row else 0.0
                    l2_jpi = l2_row["jpi"] if l2_row else ""

                    f1_e = float(g1["final_level_energy"]) if g1["final_level_energy"] else 0.0
                    f2_e = float(g2["final_level_energy"]) if g2["final_level_energy"] else 0.0

                    is_direct_1_to_2 = abs(f1_e - l2_e) < 2.5
                    is_direct_2_to_1 = abs(f2_e - l1_e) < 2.5
                    is_direct_cascade = is_direct_1_to_2 or is_direct_2_to_1
                    max_exc = max(l1_e, l2_e)

                    if is_direct_cascade:
                        cascade_type = "Direct Cascade (Prompt Coincidence)"
                        gap = 0.0
                        topo_weight = 10.0
                    elif (l1_e >= l2_e and f1_e >= l2_e) or (l2_e >= l1_e and f2_e >= l1_e):
                        gap = (f1_e - l2_e) if (l1_e >= l2_e) else (f2_e - l1_e)
                        cascade_type = f"Sequential Cascade (ΔE_gap = {gap:.0f} keV)"
                        topo_weight = 4.0 / (1.0 + gap / 600.0)
                    else:
                        gap = abs(l1_e - l2_e)
                        cascade_type = f"Same Level Scheme (Exc = {max_exc:.0f} keV)"
                        topo_weight = 1.2 / (1.0 + gap / 1200.0)

                    # Excitation energy damping factor (higher excitation -> less likely in standard decay)
                    exc_penalty = 1.0 / (1.0 + (max_exc / 3000.0) ** 1.8)

                    # Energy difference chi2 Gaussian probability
                    sig1 = max(0.2, tol1 / 2.0)
                    sig2 = max(0.2, tol2 / 2.0)
                    chi2 = (diff1 / sig1) ** 2 + (diff2 / sig2) ** 2
                    p_e = math.exp(-0.5 * chi2)

                    # Intensity factor
                    i1 = float(g1["intensity"]) if g1["intensity"] is not None else 10.0
                    i2 = float(g2["intensity"]) if g2["intensity"] is not None else 10.0
                    int_factor = math.sqrt(max(0.1, i1) * max(0.1, i2))

                    is_decay = bool(ds_meta["parent_t12_s"] or ds_meta["ds_type"] == "decay")
                    decay_boost = 1.3 if is_decay else 1.0

                    score = p_e * int_factor * topo_weight * exc_penalty * decay_boost

                    candidates.append({
                        "nuclide": ds_meta["nuid"],
                        "z": ds_meta["z"],
                        "a": ds_meta["a"],
                        "element": ds_meta["element"],
                        "dataset": ds_meta["dsid"],
                        "ds_type": ds_meta["ds_type"],
                        "parent_nuclide": ds_meta["parent_nuclide"] or ds_meta["nuid"],
                        "parent_halflife": ds_meta["parent_t12_str"] or "Prompt / In-Beam",
                        "gamma1_energy": e1_val,
                        "gamma2_energy": e2_val,
                        "diff1": round(e1_val - energy1, 3),
                        "diff2": round(e2_val - energy2, 3),
                        "level1_init": l1_e,
                        "level1_final": f1_e,
                        "level1_jpi": l1_jpi,
                        "level2_init": l2_e,
                        "level2_final": f2_e,
                        "level2_jpi": l2_jpi,
                        "max_exc": max_exc,
                        "gap": gap,
                        "cascade_type": cascade_type,
                        "is_direct_cascade": is_direct_cascade,
                        "score": round(score, 1),
                    })

        candidates.sort(key=lambda x: x["score"], reverse=True)

        if unique_nuclides:
            seen_nuclides = set()
            unique_list = []
            for c in candidates:
                if c["nuclide"] not in seen_nuclides:
                    seen_nuclides.add(c["nuclide"])
                    unique_list.append(c)
                    if len(unique_list) >= limit:
                        break
            return unique_list

        return candidates[:limit]

    def parse_fit_results_file(self, filepath: str) -> Dict[str, Any]:
        """
        Parse a python-cmat fit_results_*.txt file and extract all 1D and 2D fit lines.
        Handles both fixed-width blank-padded and legacy tab-separated files.
        """
        p = Path(filepath).expanduser()
        if not p.is_absolute():
            p_candidate = (Path.cwd() / p).resolve()
            p = p_candidate if p_candidate.exists() else p.resolve()
        else:
            p = p.resolve()

        if not p.exists():
            raise FileNotFoundError(f"Fit results file not found: {filepath}")

        fits_1d = []
        fits_2d = []

        with open(p, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line_str = line.strip()
                if not line_str or line_str.startswith("#"):
                    continue

                tokens = [t.strip() for t in re.split(r"\t+|\s{2,}", line_str) if t.strip()]

                # 1D line: 5 columns: energy(err), net_area(err), fwhm(err), chi2, peak/bg
                if len(tokens) == 5:
                    e_match = re.match(r"^([0-9\.]+)\s*(?:\(([0-9\.]+)\))?", tokens[0])
                    a_match = re.match(r"^([0-9\.]+)\s*(?:\(([0-9\.]+)\))?", tokens[1])
                    f_match = re.match(r"^([0-9\.]+)\s*(?:\(([0-9\.]+)\))?", tokens[2])
                    if e_match:
                        energy = float(e_match.group(1))
                        err = float(e_match.group(2)) if e_match.group(2) else 0.1
                        area = float(a_match.group(1)) if a_match else 0.0
                        area_err = float(a_match.group(2)) if a_match and a_match.group(2) else 0.0
                        fwhm = float(f_match.group(1)) if f_match else 0.0
                        chi2 = float(tokens[3]) if parse_float_safe(tokens[3]) else 1.0
                        pbg = float(tokens[4]) if parse_float_safe(tokens[4]) else 1.0

                        fits_1d.append({
                            "line_num": line_idx + 1,
                            "energy": energy,
                            "energy_err": err,
                            "area": area,
                            "area_err": area_err,
                            "fwhm": fwhm,
                            "chi2": chi2,
                            "pbg": pbg,
                            "raw_line": line_str,
                        })

                # 2D line: 8 columns: e1(err), e2(err), net_area(err), gamba_area(err), f1(err), f2(err), chi2, pbg
                elif len(tokens) >= 8:
                    e1_match = re.match(r"^([0-9\.]+)\s*(?:\(([0-9\.]+)\))?", tokens[0])
                    e2_match = re.match(r"^([0-9\.]+)\s*(?:\(([0-9\.]+)\))?", tokens[1])
                    a_match = re.match(r"^([0-9\.]+)\s*(?:\(([0-9\.]+)\))?", tokens[2])
                    g_match = re.match(r"^([0-9\.]+)\s*(?:\(([0-9\.]+)\))?", tokens[3])
                    if e1_match and e2_match:
                        e1 = float(e1_match.group(1))
                        e1_err = float(e1_match.group(2)) if e1_match.group(2) else 0.1
                        e2 = float(e2_match.group(1))
                        e2_err = float(e2_match.group(2)) if e2_match.group(2) else 0.1
                        area = float(a_match.group(1)) if a_match else 0.0
                        gamba = float(g_match.group(1)) if g_match else 0.0

                        fits_2d.append({
                            "line_num": line_idx + 1,
                            "energy1": e1,
                            "energy1_err": e1_err,
                            "energy2": e2,
                            "energy2_err": e2_err,
                            "area": area,
                            "gamba_area": gamba,
                            "raw_line": line_str,
                        })

        return {
            "filename": p.name,
            "filepath": str(p),
            "fits_1d": fits_1d,
            "fits_2d": fits_2d,
            "total_fits": len(fits_1d) + len(fits_2d),
        }

    def identify_fit_results_file(
        self,
        filepath: str,
        tol: float = 1.5,
        a_min: Optional[int] = None,
        a_max: Optional[int] = None,
        elements: Optional[List[str]] = None,
        min_t12_s: Optional[float] = None,
        max_t12_s: Optional[float] = None,
        dataset_type: Optional[str] = None,
        top_candidates: int = 5,
        unique_nuclides: bool = True,
        enforce_mass_cluster: bool = True,
    ) -> Dict[str, Any]:
        """
        Process an entire fit results file and return globally consistent identifications
        for all 1D and 2D fits using parsimonious isotope set-cover and mass-clustering.
        2D coincidences guide 1D photopeak assignments to find the minimal number of
        distinct #1 isotope candidates with compact mass distribution.
        """
        parsed = self.parse_fit_results_file(filepath)

        # 1. Broad Candidate Retrieval
        fits_2d_raw = []
        for fit in parsed["fits_2d"]:
            cands = self.search_2d(
                fit["energy1"],
                fit["energy2"],
                tol1=max(tol, fit["energy1_err"] * 3.0),
                tol2=max(tol, fit["energy2_err"] * 3.0),
                a_min=a_min,
                a_max=a_max,
                elements=elements,
                min_t12_s=min_t12_s,
                max_t12_s=max_t12_s,
                dataset_type=dataset_type,
                limit=40,
                unique_nuclides=False,
            )
            fits_2d_raw.append({"fit": fit, "candidates": cands})

        fits_1d_raw = []
        for fit in parsed["fits_1d"]:
            cands = self.search_1d(
                fit["energy"],
                tol=max(tol, fit["energy_err"] * 3.0),
                a_min=a_min,
                a_max=a_max,
                elements=elements,
                min_t12_s=min_t12_s,
                max_t12_s=max_t12_s,
                dataset_type=dataset_type,
                limit=60,
            )
            fits_1d_raw.append({"fit": fit, "candidates": cands})

        # 2. Identify Dominant Mass Distribution from Strongest 2D Coincidences
        mass_votes: Dict[int, float] = {}
        for f in fits_2d_raw:
            for c in f["candidates"]:
                weight = c["score"] * (4.0 if c.get("is_direct_cascade") else 0.8)
                if c["a"] is not None:
                    mass_votes[c["a"]] = mass_votes.get(c["a"], 0.0) + weight

        for f in fits_1d_raw:
            for c in f["candidates"][:3]:
                if c["a"] is not None:
                    mass_votes[c["a"]] = mass_votes.get(c["a"], 0.0) + c["score"] * 0.15

        sorted_masses = sorted(mass_votes.items(), key=lambda x: x[1], reverse=True)
        primary_mass = sorted_masses[0][0] if sorted_masses else None

        if primary_mass is not None and enforce_mass_cluster:
            cluster_weights = [w for a, w in sorted_masses if abs(a - primary_mass) <= 8]
            cluster_masses = [a for a, w in sorted_masses if abs(a - primary_mass) <= 8]
            avg_mass = sum(a * w for a, w in zip(cluster_masses, cluster_weights)) / sum(cluster_weights)
        else:
            avg_mass = None

        # 3. Seed High-Confidence 2D Anchors
        active_isotopes = set()
        for f in fits_2d_raw:
            for c in f["candidates"]:
                if c.get("is_direct_cascade"):
                    if avg_mass is None or (c["a"] is not None and abs(c["a"] - avg_mass) <= 6):
                        active_isotopes.add(c["nuclide"])
                        break

        # 4. Iterative Minimum Isotope Set Selection (Parsimony Optimization)
        def score_candidate_with_context(cand: Dict[str, Any], is_2d: bool, selected_set: set) -> float:
            base_score = float(cand.get("score", 1.0))
            a = cand.get("a")
            nuid = cand.get("nuclide")

            # Mass proximity penalty
            if avg_mass is not None and a is not None:
                da = abs(a - avg_mass)
                # Gaussian decay with sigma = 6.0 mass units
                mass_factor = math.exp(- (da ** 2) / (2.0 * (6.0 ** 2)))
                mass_factor = max(0.0001, mass_factor)
            else:
                mass_factor = 1.0

            # Parsimony bonus: prefer explaining lines with isotopes already chosen in minimum set
            parsimony_factor = 1.0
            if selected_set and nuid in selected_set:
                parsimony_factor = 5.0
            elif nuid in active_isotopes:
                parsimony_factor = 3.0

            direct_bonus = 1.4 if (is_2d and cand.get("is_direct_cascade")) else 1.0
            return base_score * mass_factor * parsimony_factor * direct_bonus

        chosen_isotopes = set(active_isotopes)
        for _ in range(3):
            for f in fits_2d_raw + fits_1d_raw:
                for c in f["candidates"]:
                    c["temp_score"] = score_candidate_with_context(
                        c, is_2d=("gamma2_energy" in c), selected_set=chosen_isotopes
                    )
                f["candidates"].sort(key=lambda x: x["temp_score"], reverse=True)
                if f["candidates"]:
                    top_cand = f["candidates"][0]
                    if avg_mass is None or (top_cand["a"] is not None and abs(top_cand["a"] - avg_mass) <= 8):
                        chosen_isotopes.add(top_cand["nuclide"])

        # 5. Final Re-ranking & Formatting for 2D Coincidences
        id_2d = []
        for f in fits_2d_raw:
            fit = f["fit"]
            for c in f["candidates"]:
                c["global_score"] = round(
                    score_candidate_with_context(c, is_2d=True, selected_set=chosen_isotopes), 1
                )
                c["score"] = c["global_score"]
            f["candidates"].sort(key=lambda x: x["global_score"], reverse=True)

            # Deduplicate by distinct nuclide if requested
            seen_nuclides = set()
            filtered_cands = []
            for c in f["candidates"]:
                if unique_nuclides:
                    if c["nuclide"] not in seen_nuclides:
                        seen_nuclides.add(c["nuclide"])
                        filtered_cands.append(c)
                else:
                    filtered_cands.append(c)
                if len(filtered_cands) >= top_candidates:
                    break

            id_2d.append({
                "fit": fit,
                "candidates": filtered_cands,
                "best_match": filtered_cands[0] if filtered_cands else None,
            })

        # 6. Final Re-ranking & Formatting for 1D Photopeaks
        id_1d = []
        for f in fits_1d_raw:
            fit = f["fit"]
            for c in f["candidates"]:
                c["global_score"] = round(
                    score_candidate_with_context(c, is_2d=False, selected_set=chosen_isotopes), 1
                )
                c["score"] = c["global_score"]
            f["candidates"].sort(key=lambda x: x["global_score"], reverse=True)

            seen_nuclides = set()
            filtered_cands = []
            for c in f["candidates"]:
                if unique_nuclides:
                    if c["nuclide"] not in seen_nuclides:
                        seen_nuclides.add(c["nuclide"])
                        filtered_cands.append(c)
                else:
                    filtered_cands.append(c)
                if len(filtered_cands) >= top_candidates:
                    break

            id_1d.append({
                "fit": fit,
                "candidates": filtered_cands,
                "best_match": filtered_cands[0] if filtered_cands else None,
            })

        return {
            "file": parsed["filename"],
            "filepath": parsed["filepath"],
            "dominant_mass": round(avg_mass, 1) if avg_mass else None,
            "parsimonious_isotopes": sorted(list(chosen_isotopes)),
            "results_1d": id_1d,
            "results_2d": id_2d,
        }


# ==============================================================================
# CLI Terminal Reports & Formatting
# ==============================================================================

def print_1d_search_report(energy: float, results: List[Dict[str, Any]]):
    """Print clean terminal report for 1D single-energy search."""
    bar = "═" * 105
    print(f"\n{bar}")
    print(f" ENSDF Isotope Identification: 1D Photopeak E_gamma = {energy:.2f} keV ({len(results)} matches)")
    print(bar)
    if not results:
        print("  No matching transitions found under the current constraints.")
        print(f"{bar}\n")
        return

    hdr = f"{'#':<3} {'Nuclide':<9} {'E_ensdf (keV)':<14} {'Delta_E':<10} {'Intensity':<10} {'Parent / Decay Mode':<28} {'Half-Life':<16} {'Score':<7}"
    print(hdr)
    print("─" * 105)
    for idx, r in enumerate(results, 1):
        diff_str = f"{r['energy_diff']:+.3f}"
        int_str = f"{r['intensity']:.1f}" if r['intensity'] is not None else "-"
        parent_str = f"{r['parent_nuclide']} ({r['ds_type']})" if r['parent_nuclide'] != r['nuclide'] else r['dataset'][:26]
        print(
            f"{idx:<3} {r['nuclide']:<9} {r['gamma_energy']:<14.3f} {diff_str:<10} {int_str:<10} "
            f"{parent_str:<28} {r['parent_halflife']:<16} {r['score']:<7.1f}"
        )
    print(f"{bar}\n")


def print_2d_search_report(energy1: float, energy2: float, results: List[Dict[str, Any]]):
    """Print clean terminal report for 2D coincidence search listing top likely candidate isotopes."""
    bar = "═" * 120
    print(f"\n{bar}")
    print(f" ENSDF Isotope Identification: 2D Coincidence ({energy1:.2f} x {energy2:.2f} keV) [Top {len(results)} Candidates]")
    print(bar)
    if not results:
        print("  No coincident pairs found under the current constraints.")
        print(f"{bar}\n")
        return

    hdr = f"{'#':<3} {'Nuclide':<9} {'Score':<7} {'Cascade Topology':<35} {'Level Transitions (keV)':<26} {'Parent (T1/2)':<24} {'ΔE1, ΔE2':<12}"
    print(hdr)
    print("─" * 120)
    for idx, r in enumerate(results, 1):
        diff_str = f"{r['diff1']:+.2f}, {r['diff2']:+.2f}"
        levels_str = f"L1({r['level1_init']:.0f}→{r['level1_final']:.0f}) L2({r['level2_init']:.0f}→{r['level2_final']:.0f})"
        parent_info = f"{r['parent_nuclide']} ({r['parent_halflife'][:12]})"
        print(
            f"{idx:<3} {r['nuclide']:<9} {r['score']:<7.1f} {r['cascade_type'][:34]:<35} "
            f"{levels_str:<26} {parent_info:<24} {diff_str:<12}"
        )
    print(f"{bar}\n")


def print_file_identification_report(report: Dict[str, Any]):
    """Print terminal report for full fit results file."""
    bar = "═" * 115
    print(f"\n{bar}")
    print(f" Automatic Isotope Identification: {report['file']}")
    if report.get("dominant_mass") is not None:
        isotopes_str = ", ".join(report.get("parsimonious_isotopes", []))
        print(f" Dominant Mass Center: A ≈ {report['dominant_mass']} | Minimal Isotope Set: [{isotopes_str}]")
    print(bar)

    if report["results_2d"]:
        print(f"\n[+] 2D Coincidence Fits ({len(report['results_2d'])} peaks):")
        for idx, item in enumerate(report["results_2d"], 1):
            fit = item["fit"]
            candidates = item.get("candidates", [])
            print(f"  [{idx}] Coincidence ({fit['energy1']:.1f} x {fit['energy2']:.1f} keV) Net Area: {fit['area']:.1f}:")
            if not candidates:
                print("      --> No candidate isotope found under current constraints.")
                continue

            for c_idx, c in enumerate(candidates, 1):
                badge = "★ TOP MATCH" if c_idx == 1 else f"  #{c_idx} Candidate"
                print(
                    f"      {badge:<14}: {c['nuclide']:<7} (Parent: {c['parent_nuclide']}, T1/2: {c['parent_halflife']}) | "
                    f"Score: {c['score']:<5.1f} | {c['cascade_type']}"
                )
                print(
                    f"                       Transitions: {c['gamma1_energy']:.2f} & {c['gamma2_energy']:.2f} keV "
                    f"(Δ: {c['diff1']:+.2f}, {c['diff2']:+.2f} keV) | Level Scheme: {c['level1_init']:.0f}→{c['level1_final']:.0f} & {c['level2_init']:.0f}→{c['level2_final']:.0f} keV"
                )

    if report["results_1d"]:
        print(f"\n[+] 1D Photopeak Fits ({len(report['results_1d'])} peaks):")
        for idx, item in enumerate(report["results_1d"], 1):
            fit = item["fit"]
            best = item["best_match"]
            print(f"  [{idx}] 1D Peak {fit['energy']:.2f} keV (Area: {fit['area']:.1f}):")
            if best:
                print(f"      --> Best Match: {best['nuclide']} ({best['parent_nuclide']}) | Score: {best['score']} | E_ensdf: {best['gamma_energy']:.2f} keV (diff: {best['energy_diff']:+.2f})")
                print(f"          Dataset: {best['dataset']} | T1/2: {best['parent_halflife']}")
            else:
                print("      --> No candidate isotope found under current constraints.")

    print(f"\n{bar}\n")


# ==============================================================================
# Main CLI Entry Point
# ==============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Automated Nuclear Isotope Search Engine & ENSDF Database Toolkit for python-cmat."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=None,
        help="Optional fit_results_*.txt file to analyze and identify automatically",
    )
    parser.add_argument(
        "--build-db",
        nargs="?",
        const=DEFAULT_ENSDF_DIR,
        default=None,
        metavar="ENSDF_DIR",
        help=f"Build/index local SQLite database from raw ENSDF directory (default: {DEFAULT_ENSDF_DIR})",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=DEFAULT_DB_FILENAME,
        help=f"Path to SQLite database file (default: {DEFAULT_DB_FILENAME})",
    )
    parser.add_argument(
        "-g", "--gamma",
        type=float,
        default=None,
        metavar="ENERGY_KEV",
        help="Single gamma-ray energy in keV to identify",
    )
    parser.add_argument(
        "-c", "--coinc",
        nargs=2,
        type=float,
        default=None,
        metavar=("E1_KEV", "E2_KEV"),
        help="2D coincidence pair (E1, E2) in keV to identify",
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=5,
        metavar="N",
        help="Maximum number of top likely isotope candidates to return for 2D coincidences (default: 5)",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Do not deduplicate by unique nuclide symbol; show all dataset matches up to top N",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1.5,
        help="Energy search tolerance in keV (default: 1.5 keV)",
    )
    parser.add_argument(
        "--a-min",
        type=int,
        default=None,
        help="Minimum mass number A constraint (e.g. 50)",
    )
    parser.add_argument(
        "--a-max",
        type=int,
        default=None,
        help="Maximum mass number A constraint (e.g. 70)",
    )
    parser.add_argument(
        "-e", "--element",
        nargs="+",
        default=None,
        help="Filter by specific element symbols (e.g. Ni Co Fe)",
    )
    parser.add_argument(
        "--min-t12",
        type=str,
        default=None,
        help="Minimum parent half-life constraint (e.g. '1s', '10m', '1h', '30d', '1y')",
    )
    parser.add_argument(
        "--max-t12",
        type=str,
        default=None,
        help="Maximum parent half-life constraint (e.g. '5y', '100y')",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Display database status, nuclide counts, and statistics",
    )

    args = parser.parse_args()

    # 1. Build Database action
    if args.build_db is not None:
        build_ensdf_database(args.build_db, db_path=args.db, verbose=True)
        sys.exit(0)

    engine = ENSDFSearchEngine(db_path=args.db)

    # 2. Check Database Status
    if args.status or (args.file is None and args.gamma is None and args.coinc is None):
        stats = engine.get_stats()
        print("\n" + "═" * 60)
        print(" python-cmat ENSDF Search Engine Status")
        print("═" * 60)
        if stats["available"]:
            print(f" Database Path:    {stats['db_path']}")
            print(f" Database Size:    {stats['size_mb']} MB")
            print(f" Total Nuclides:   {stats['nuclides']:,}")
            print(f" Total Datasets:   {stats['datasets']:,}")
            print(f" Total Levels:     {stats['levels']:,}")
            print(f" Total Gammas:     {stats['gammas']:,}")
            print(" Status:           READY")
        else:
            print(f" Database File:    {stats['db_path']} (NOT FOUND)")
            print(f" Status:           UNINITIALIZED")
            print("\n Build the database with:")
            print(f"   python3 ensdf_search.py --build-db {DEFAULT_ENSDF_DIR}")
        print("═" * 60 + "\n")
        if not args.file and args.gamma is None and args.coinc is None:
            sys.exit(0)

    min_t12_s = parse_human_duration(args.min_t12)
    max_t12_s = parse_human_duration(args.max_t12)

    # 3. 2D Coincidence Search
    if args.coinc is not None:
        e1, e2 = args.coinc
        res = engine.search_2d(
            e1, e2, tol1=args.tol, tol2=args.tol,
            a_min=args.a_min, a_max=args.a_max, elements=args.element,
            min_t12_s=min_t12_s, max_t12_s=max_t12_s,
            limit=args.top,
            unique_nuclides=not args.all_datasets,
        )
        print_2d_search_report(e1, e2, res)
        sys.exit(0)

    # 4. 1D Single-Gamma Search
    if args.gamma is not None:
        res = engine.search_1d(
            args.gamma, tol=args.tol,
            a_min=args.a_min, a_max=args.a_max, elements=args.element,
            min_t12_s=min_t12_s, max_t12_s=max_t12_s,
            limit=args.top,
        )
        print_1d_search_report(args.gamma, res)
        sys.exit(0)

    # 5. Full Fit Results File Identification
    if args.file is not None:
        rep = engine.identify_fit_results_file(
            args.file, tol=args.tol,
            a_min=args.a_min, a_max=args.a_max, elements=args.element,
            min_t12_s=min_t12_s, max_t12_s=max_t12_s,
            top_candidates=args.top,
            unique_nuclides=not args.all_datasets,
        )
        print_file_identification_report(rep)
        sys.exit(0)


if __name__ == "__main__":
    main()

