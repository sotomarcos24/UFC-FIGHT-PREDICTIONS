import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

FILES = {
    "events":   "ufc_event_details.csv",
    "results":  "ufc_fight_results.csv",
    "stats":    "ufc_fight_stats.csv",
    "fighters": "ufc_fighter_tott.csv",
}

STATS = [
    "sig_str_landed", "sig_str_attempted",
    "total_str_landed", "total_str_attempted",
    "td_landed", "td_attempted", "ctrl_seconds", "kd", "sub_att",
    "sig_str_landed_opp", "sig_str_attempted_opp",
    "td_landed_opp", "td_attempted_opp", "ctrl_seconds_opp", "kd_opp",
    "total_time",
]

FIGHTER_FEATS = [
        "prev_fights", "prev_wins", "win_rate",
        "slpm", "sapm", "str_acc", "str_def",
        "td_avg", "td_acc", "td_def", "sub_avg", "kd_avg", "ctrl_pct",
        "height_cm", "reach_cm", "age",
    ]

CONTEXT = ["weight_class", "is_title", "is_women", "scheduled_rounds"]

EXTRA_EVENT_DATES = {"UFC - Road to UFC 4.6": "August 22, 2025"}

ID_COLS = ["DATE", "EVENT", "BOUT", "FIGHTER", "OPPONENT"]
TARGET = "win"

DIFF_FEATS = [f"{c}_diff" for c in FIGHTER_FEATS]
EXTRA_FEATS = ["STANCE", "STANCE_b", "same_stance", "prev_fights", "prev_fights_b"]

FEATURES = DIFF_FEATS + CONTEXT + EXTRA_FEATS

MIN_DIVISION_SHARE = 0.2


def build_matchup_features(snapshot, name_a, name_b, context=None):
    """Build one model row (same columns as FEATURES) for a fight between two fighters."""
    fighters = snapshot.set_index("FIGHTER")

    missing = [n for n in (name_a, name_b) if n not in fighters.index]
    if missing:
        raise KeyError(f"Fighter(s) not found: {', '.join(missing)}")
    if name_a == name_b:
        raise ValueError("A fighter cannot fight himself")

    a = fighters.loc[name_a]
    b = fighters.loc[name_b]

    shared = shared_divisions(snapshot, name_a, name_b)
    if not shared:
        raise ValueError(
            f"{name_a} ({', '.join(a['divisions'])}) and {name_b} ({', '.join(b['divisions'])}) "
            "do not compete in a common weight class"
        )

    context = dict(context or {})
    if "weight_class" in context and context["weight_class"] not in shared:
        raise ValueError(f"Weight class must be one of: {', '.join(shared)}")
    default_wc = a["weight_class"] if a["weight_class"] in shared else shared[0]

    row = {}
    for c in FIGHTER_FEATS:
        row[f"{c}_diff"] = float(a[c] - b[c])

    row["STANCE"] = a["STANCE"]
    row["STANCE_b"] = b["STANCE"]
    row["same_stance"] = int(a["STANCE"] == b["STANCE"])
    row["prev_fights"] = int(a["prev_fights"])
    row["prev_fights_b"] = int(b["prev_fights"])

    defaults = {
        "weight_class": default_wc,
        "is_title": 0,
        "is_women": int(a["is_women"]),
        "scheduled_rounds": 3,
    }
    row.update({**defaults, **context})

    assert set(row) == set(FEATURES), f"Feature mismatch: {set(row) ^ set(FEATURES)}"
    return row


def matchup_warnings(snapshot, name_a, name_b, today=None):
    today = pd.Timestamp.today() if today is None else pd.Timestamp(today)
    fighters = snapshot.set_index("FIGHTER")
    a, b = fighters.loc[name_a], fighters.loc[name_b]
    warnings = []

    for name, f in ((name_a, a), (name_b, b)):
        years = (today - f["last_fight_date"]).days / 365.25
        if years > 2:
            warnings.append(f"{name} has not fought in the UFC for {years:.1f} years")
        if f["prev_fights"] < 3:
            warnings.append(f"{name} has only {int(f['prev_fights'])} UFC fights: limited data")
    return warnings



def compute_ratios(df):
    mins = df["prev_total_time"] / 60
    mins = mins.where(mins >= 5)

    df["slpm"]     = df["prev_sig_str_landed"] / mins
    df["sapm"]     = df["prev_sig_str_landed_opp"] / mins
    df["str_acc"]  = df["prev_sig_str_landed"] / df["prev_sig_str_attempted"]
    df["str_def"]  = 1 - df["prev_sig_str_landed_opp"] / df["prev_sig_str_attempted_opp"]
    df["ctrl_pct"] = df["prev_ctrl_seconds"] / df["prev_ctrl_time"]
    df["win_rate"] = df["prev_wins"] / df["prev_fights"]
    df["td_avg"]   = df["prev_td_landed"] / mins * 15
    df["td_acc"]   = df["prev_td_landed"] / df["prev_td_attempted"]
    df["td_def"]   = 1 - df["prev_td_landed_opp"] / df["prev_td_attempted_opp"]
    df["sub_avg"]  = df["prev_sub_att"] / mins * 15
    df["kd_avg"]   = df["prev_kd"] / mins * 15

    return df.replace([np.inf, -np.inf], np.nan)


def load_raw(data_dir=DATA_DIR):
    raw = {}
    for name, filename in FILES.items():
        raw[name] = pd.read_csv(Path(data_dir) / filename)
    return raw


def clean_events(df): 
    df = df.drop(["URL", "LOCATION"], axis=1)
    return df


def clean_fight_results(df): 
    df = df[df["TIME FORMAT"].isin(["5 Rnd (5-5-5-5-5)", "3 Rnd (5-5-5)"])]
    df["scheduled_rounds"] = df["TIME FORMAT"].str[0].astype(int)
    df["total_time"] = (df["ROUND"].astype(int) - 1) * 5 * 60 + (df["TIME"].str[0].astype(int) * 60 + df["TIME"].str[2:].astype(int))

    parts = df["BOUT"].str.strip().str.split(" vs. ", expand=True)
    df["fighter_a"] = parts[0].str.strip()
    df["fighter_b"] = parts[1].str.strip()
    df["a_wins"] = df["OUTCOME"].str.strip().map({"W/L": 1, "L/W": 0})

    wc = df["WEIGHTCLASS"].str.strip()

    df["is_women"] = wc.str.contains("Women").astype(int)
    df["is_title"] = wc.str.contains("Title").astype(int)

    CLASSES = [
        "Light Heavyweight", "Super Heavyweight", "Heavyweight",  
        "Middleweight", "Welterweight", "Lightweight", "Featherweight",
        "Bantamweight", "Flyweight", "Strawweight", "Catch Weight",
    ]

    def normalize_weight_class(text):
        for c in CLASSES:
            if c in text:
                return c
        return "Other"

    df["weight_class"] = wc.map(normalize_weight_class)

    df = df[~df["weight_class"].isin(["Super Heavyweight", "Other"])]

    df = df.drop(
        columns=[
            "WEIGHTCLASS", "URL", "DETAILS", "REFEREE", "TIME", "TIME FORMAT", "ROUND", "OUTCOME"
        ]
    )
    return df


def clean_fight_stats(df): 
    df = df.dropna(subset=["FIGHTER"])
    df["FIGHTER"] = df["FIGHTER"].str.strip()

    OF_COLS = {
        "SIG.STR.": "sig_str", "TOTAL STR.": "total_str", "TD": "td",
        "HEAD": "head", "BODY": "body", "LEG": "leg",
        "DISTANCE": "distance", "CLINCH": "clinch", "GROUND": "ground",
    }

    for col, name in OF_COLS.items():
        parts = df[col].str.strip().str.split(" of ", expand=True)
        df[f"{name}_landed"] = parts[0].str.strip().astype(int)
        df[f"{name}_attempted"] = parts[1].str.strip().astype(int)

    for col, name in OF_COLS.items():
        df = df.drop([col], axis=1)

    ctrl = df["CTRL"].str.strip().replace("--", np.nan)
    t = ctrl.str.split(":", expand=True)
    df["ctrl_seconds"] = t[0].astype(float) * 60 + t[1].astype(float)

    df = df.rename(columns={"KD": "kd", "SUB.ATT": "sub_att", "REV.": "rev"})
    df = df.drop(columns=["CTRL", "ROUND", "SIG.STR. %", "TD %"])

    return df


def clean_fighters(df):
    df["FIGHTER"] = df["FIGHTER"].str.strip()

    reach = df["REACH"].str.strip().str.replace('"', "", regex=False)
    df["reach_cm"] = pd.to_numeric(reach, errors="coerce") * 2.54

    df = df.drop(columns=["WEIGHT", "URL"])

    df["HEIGHT"] = df["HEIGHT"].str.strip().replace("--", np.nan)

    parts = df["HEIGHT"].str.strip().str.replace('"', "", regex=False).str.split("'", expand=True)

    feet = pd.to_numeric(parts[0].str.strip(), errors="coerce")
    inch = pd.to_numeric(parts[1].str.strip(), errors="coerce")
    df["height_cm"] = (feet * 12 + inch) * 2.54

    df = df.drop(columns=["HEIGHT", "REACH"])
    df["DOB"] = pd.to_datetime(df["DOB"].str.strip(), format="%b %d, %Y", errors="coerce")
    df["STANCE"] = df["STANCE"].fillna("Unknown")

    df["STANCE"] = df["STANCE"].replace({"Open Stance": "Unknown", "Sideways": "Unknown"})

    return df


def build_fighter_rows(details, results, stats): 
    df_fight = (
        stats
        .groupby(["EVENT", "BOUT", "FIGHTER"], as_index=False)
        .sum(min_count=1)
    )

    KEEP = ["EVENT", "BOUT", "FIGHTER",
        "sig_str_landed", "sig_str_attempted",
        "total_str_landed", "total_str_attempted",
        "td_landed", "td_attempted", "ctrl_seconds",
        "kd", "sub_att"]
    
    df_fight = df_fight[KEEP]

    opp = df_fight.rename(columns={"FIGHTER": "OPPONENT"})
    df_full = df_fight.merge(opp, on=["EVENT", "BOUT"], suffixes=("", "_opp"))
    df_full = df_full[df_full["FIGHTER"] != df_full["OPPONENT"]]

    for df in (df_full, results):
        df["EVENT"] = df["EVENT"].str.strip()
        df["BOUT"] = df["BOUT"].str.strip()

    details["EVENT"] = details["EVENT"].str.strip()

    details = pd.concat([
        details,
        pd.DataFrame({"EVENT": list(EXTRA_EVENT_DATES), "DATE": list(EXTRA_EVENT_DATES.values())}),
    ], ignore_index=True)

    df = df_full.merge(results, on=["EVENT", "BOUT"], how="inner")

    df["win"] = np.where(
        df["FIGHTER"] == df["fighter_a"],  
        df["a_wins"],                       
        1 - df["a_wins"],                   
    )

    df = df.merge(details[["EVENT", "DATE"]], on="EVENT", how="inner")
    df["DATE"] = pd.to_datetime(df["DATE"])
    df = df.sort_values(["DATE", "EVENT", "BOUT"]).reset_index(drop=True)

    df = df.drop(columns=["fighter_a", "fighter_b", "a_wins"])

    return df


def add_history_features(df, fighter_tott):
    dup_names = fighter_tott.loc[fighter_tott["FIGHTER"].duplicated(keep=False), "FIGHTER"].unique()
    df = df[~df["FIGHTER"].isin(dup_names) & ~df["OPPONENT"].isin(dup_names)]
    fighter_tott = fighter_tott[~fighter_tott["FIGHTER"].isin(dup_names)]

    n_before = len(df)
    df = df.merge(
        fighter_tott[["FIGHTER", "height_cm", "reach_cm", "STANCE", "DOB"]],
        on="FIGHTER", how="left",
    )

    df["STANCE"] = df["STANCE"].fillna("Unknown")

    assert len(df) == n_before

    df["age"] = (df["DATE"] - df["DOB"]).dt.days / 365.25
    df = df.drop(columns=["DOB"])

    df = df.sort_values(["DATE", "EVENT", "BOUT"]).reset_index(drop=True)

    g = df.groupby("FIGHTER")

    df["prev_fights"] = g.cumcount()
    w = df["win"].fillna(0)
    df["prev_wins"] = w.groupby(df["FIGHTER"]).cumsum() - w

    for col in STATS:
            x = df[col].fillna(0)
            df[f"prev_{col}"] = x.groupby(df["FIGHTER"]).cumsum() - x
    
    ctrl_known_time = df["total_time"].where(df["ctrl_seconds"].notna(), 0)
    df["prev_ctrl_time"] = ctrl_known_time.groupby(df["FIGHTER"]).cumsum() - ctrl_known_time

    return compute_ratios(df)


def build_fight_dataset(df):
    side_a = df[["EVENT", "BOUT", "DATE", "FIGHTER", "OPPONENT", "STANCE", "win"]
                + CONTEXT + FIGHTER_FEATS]

    side_b = df[["EVENT", "BOUT", "FIGHTER", "STANCE"] + FIGHTER_FEATS].rename(
        columns={"FIGHTER": "OPPONENT", "STANCE": "STANCE_b",
                **{c: f"{c}_b" for c in FIGHTER_FEATS}}
    )

    fights = side_a.merge(side_b, on=["EVENT", "BOUT", "OPPONENT"], how="inner")
    assert len(fights) == len(df)

    for c in FIGHTER_FEATS:
        fights[f"{c}_diff"] = fights[c] - fights[f"{c}_b"]
    fights["same_stance"] = (fights["STANCE"] == fights["STANCE_b"]).astype(int)

    DIFF_FEATS = [f"{c}_diff" for c in FIGHTER_FEATS]
    FEATURES = DIFF_FEATS + CONTEXT + ["STANCE", "STANCE_b", "same_stance",
                                    "prev_fights", "prev_fights_b"]

    model_df = fights.dropna(subset=["win"]).copy()
    model_df = model_df[["DATE", "EVENT", "BOUT", "FIGHTER", "OPPONENT"] + FEATURES + ["win"]]
    model_df["win"] = model_df["win"].astype(int)

    return model_df


def fighter_divisions(df, min_share=MIN_DIVISION_SHARE):
    """Divisions where each fighter has at least `min_share` of his fights, plus his latest one."""
    df = df.sort_values("DATE")
    real = df[df["weight_class"] != "Catch Weight"]
    shares = real.groupby("FIGHTER")["weight_class"].value_counts(normalize=True)
    frequent = shares[shares >= min_share].reset_index().groupby("FIGHTER")["weight_class"].apply(set)
    latest = real.groupby("FIGHTER")["weight_class"].last()

    divisions = {}
    for name in df["FIGHTER"].unique():
        divs = set(frequent.get(name, set()))
        if name in latest.index:
            divs.add(latest[name])
        if not divs:   # only catch-weight fights: use the last class seen
            divs.add(df.loc[df["FIGHTER"] == name, "weight_class"].iloc[-1])
        divisions[name] = sorted(divs)
    return pd.Series(divisions, name="divisions")


def shared_divisions(snapshot, name_a, name_b):
    fighters = snapshot.set_index("FIGHTER")
    return sorted(set(fighters.loc[name_a, "divisions"]) & set(fighters.loc[name_b, "divisions"]))


def build_fighter_snapshot(df, today=None):
    today = pd.Timestamp.today().normalize() if today is None else pd.Timestamp(today)
    df = df.sort_values("DATE").copy()
    df["ctrl_known_time"] = df["total_time"].where(df["ctrl_seconds"].notna(), 0)

    g = df.groupby("FIGHTER")
    snap = g[STATS + ["ctrl_known_time"]].sum()                 
    snap = snap.rename(columns={c: f"prev_{c}" for c in STATS})
    snap = snap.rename(columns={"ctrl_known_time": "prev_ctrl_time"})
    snap["prev_fights"] = g.size()
    snap["prev_wins"] = g["win"].sum()

    last = g.tail(1).set_index("FIGHTER")                        
    snap = snap.join(last[["DATE", "weight_class", "is_women", "height_cm", "reach_cm", "STANCE", "age"]])
    snap["age"] = snap["age"] + (today - snap["DATE"]).dt.days / 365.25
    snap = snap.rename(columns={"DATE": "last_fight_date"})

    snap = snap.join(fighter_divisions(df))

    return compute_ratios(snap).reset_index()


def build_datasets(data_dir=DATA_DIR):
    raw = load_raw(data_dir)
    rows = build_fighter_rows(
        clean_events(raw["events"]),
        clean_fight_results(raw["results"]),
        clean_fight_stats(raw["stats"]),
    )
    history = add_history_features(rows, clean_fighters(raw["fighters"]))
    model_df = build_fight_dataset(history)
    snapshot = build_fighter_snapshot(history)
    return model_df, snapshot