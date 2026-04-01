# -*- coding: utf-8 -*-
"""
Federated Learning for ESG Risk Assessment in Supply Chains
Design Science Research — Two-Step System Demonstration

Step 1 (Low-Frequency):  FL Training — FedAvg with rolling baseline (Plan B)
Step 2 (High-Frequency): Risk Scoring & Propagation — fixed baseline (Plan A)

Algorithm: FedAvg (Federated Averaging)
Local Model: Logistic Regression on z-score normalized ESG metrics
Privacy: SHA-256 parameter hashing (simulation)
Dimensions: Environmental (E), Social (S) — Governance excluded

@author: lulux

===============================================================================
HANDLING HETEROGENEOUS ESG VARIABLE TYPES, DIRECTIONALITY, AND BENCHMARKING
===============================================================================

PROBLEM:
    ESG raw data contains heterogeneous variable types and directions:

    Type 1 — Continuous, higher = higher risk (default direction):
        e.g., GHG Emissions (tons CO2e), Water Withdrawal (megalitres),
        Waste Generated (tons), Turnover Rate (%), Pay Gap (ratio)

    Type 2 — Continuous, higher = LOWER risk (inverted direction):
        e.g., Renewable Energy Share (%), Recycling Rate (%),
        Women on Board (%), Training Hours (hours/employee),
        Community Investment ($), Supplier Diversity (%)

    Type 3 — Binary / Categorical:
        e.g., Human Rights Policy (Yes/No), ISO 14001 Certified (Yes/No),
        Whistleblower Program (Yes/No/Partial), CDP Grade (A to F)

    Without an external absolute benchmark (e.g., industry standard, regulatory
    threshold), there is no way to say "X tons of CO2 is high risk" in absolute
    terms. Different industries, company sizes, and reporting frameworks make
    absolute thresholds impractical for cross-entity comparison.

SOLUTION — Three-stage pipeline:

    Stage 1. VALUE PARSING (parse_value):
        All raw types are converted to a common numeric scale:
        - Continuous values: parsed as-is (with %, $, comma cleaning)
        - Yes/No/Partial: mapped to 1.0 / 0.0 / 0.5
        - CDP letter grades: mapped to 0.0 (F) — 0.95 (A)
        - Ratios (e.g., "85:1"): parsed as the numeric component
        This ensures all metrics are on a numeric scale before normalization.

    Stage 2. DIRECTION ALIGNMENT (is_good_metric + normalize_and_label):
        Each indicator is classified by direction using keyword matching:
        - GOOD_METRIC_KEYWORDS: "women", "recycled", "renewable", "training
          hours", "anti-corruption", etc. → higher value = lower risk
        - Default (no keyword match): higher value = higher risk

        After z-score normalization:  z = (value - mean) / std
        Direction is unified:
        - For "good" metrics:  risk_z = -z  (flip sign, so high original → low risk_z)
        - For "risk" metrics:  risk_z =  z  (keep sign, high original → high risk_z)

        Result: risk_z > 0 always means "above-average risk" regardless of
        the original metric's direction or unit.

    Stage 3. PEER-RELATIVE BENCHMARKING (z-score as implicit benchmark):
        Instead of an absolute threshold, we use the cross-entity distribution
        (mean and std across all 7 supply chain entities) as the benchmark:
        - risk_z > 0 → above-peer-average risk → label = 1 (high risk)
        - risk_z ≤ 0 → at or below peer average → label = 0 (low risk)

        This is a RELATIVE benchmark: each firm is scored against its supply
        chain peers, not against an external standard.

        Implication: if ALL firms have high emissions, the below-average ones
        are still labeled "low risk". This is acceptable for our use case
        because the FL system learns RELATIVE risk patterns within a supply
        chain, and the risk propagation mechanism (Step 2) captures how
        deviations from the peer baseline amplify through the chain.

        The logistic regression then learns: given a risk_z value and the
        ESG sub-category (one-hot encoded), what is P(high_risk)?
        This mapping generalises across firms because all inputs are on the
        same normalised, direction-aligned scale.

    TWO-STEP BASELINE STRATEGY:
        Step 1 (FL Training) uses ROLLING baseline (Plan B):
            mean/std are recomputed from the current training data.
            This lets the model learn from the most recent peer distribution.

        Step 2 (Risk Scoring) uses FIXED baseline (Plan A):
            mean/std are frozen from the Step 1 training data (M0).
            This ensures that risk score changes between M0→M1→M2→M3
            reflect ACTUAL data changes, not baseline drift.
            e.g., if Nutrien's GHG doubles in M1, its risk_z increases
            purely due to the emission increase, not because the peer
            mean shifted.

RISK PROPAGATION VERIFICATION (M0 → M1, Nutrien GHG Doubled):
    The upstream-to-downstream propagation chain works as follows:

    Nutrien (Tier 2):   R_local  +0.0304  (direct impact of GHG doubling)
      ↓ α(Nutrien→Goodyear) = 0.20
    Goodyear (Tier 1):  R_total  +0.0061  (= 0.20 × 0.0304, attenuated)
      ↓ α(Goodyear→Mattel) = 0.12
    Mattel (Focal):     R_total  +0.0007  (= 0.12 × 0.0061, further attenuated)
      ↓ α(Mattel→Walmart) = 0.01, α(Mattel→Target) = 0.02
    Walmart (Down):     R_total  ~0.0000  (= 0.01 × 0.0007, negligible)
    Target (Down):      R_total  ~0.0000  (= 0.02 × 0.0007, negligible)

    Key observations:
    - Risk propagation IS reaching all downstream entities including Mattel.
    - The attenuation through two tiers is multiplicative: 0.20 × 0.12 = 0.024,
      so only ~2.4% of Nutrien's local risk change reaches Mattel's R_total.
    - This is by design: the focal firm's own ESG performance (β=0.70) dominates,
      while upstream supply chain risk provides early warning signals.
    - Downstream entities (Walmart, Target) are affected by Mattel but NOT
      vice versa, ensuring one-directional risk flow.
===============================================================================
"""

import numpy as np
import pandas as pd
import hashlib
import os
import warnings
from copy import deepcopy
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings('ignore')

# =============================================================================
# SUPPLY CHAIN CONFIGURATION
# =============================================================================

SUPPLY_CHAIN = {
    'Nutrien':          {'tier': 'Tier 2',      'role': 'Fertilizer Supplier',    'upstream': []},
    'Deere':            {'tier': 'Tier 2',      'role': 'Agricultural Equipment', 'upstream': []},
    'Goodyear':         {'tier': 'Tier 1',      'role': 'Rubber Factory',         'upstream': ['Nutrien', 'Deere']},
    'Sherwin-Williams': {'tier': 'Tier 1',      'role': 'Paint Manufacturer',     'upstream': []},
    'Mattel':           {'tier': 'Focal',       'role': 'Toy Manufacturer',       'upstream': ['Goodyear', 'Sherwin-Williams']},
    'Walmart':          {'tier': 'Downstream',  'role': 'Retailer 1',            'upstream': ['Mattel']},
    'Target':           {'tier': 'Downstream',  'role': 'Retailer 2',            'upstream': ['Mattel']},
}

PROPAGATION_WEIGHTS = {
    'Nutrien':          {'beta': 1.0, 'alpha': {}},
    'Deere':            {'beta': 1.0, 'alpha': {}},
    'Goodyear': {
        'beta': 0.5,
        'alpha': {'Nutrien': 0.4 * 0.5, 'Deere': 0.6 * 0.5}
    },
    'Sherwin-Williams': {'beta': 1.0, 'alpha': {}},
    'Mattel': {
        'beta': 0.7,
        'alpha': {'Goodyear': 0.4 * 0.3, 'Sherwin-Williams': 0.6 * 0.3}
    },
    'Walmart': {
        'beta': 0.99,
        'alpha': {'Mattel': 0.01}
    },
    'Target': {
        'beta': 0.98,
        'alpha': {'Mattel': 0.02}
    },
}

FL_ROUNDS = 10

COMPANY_COLUMNS = {
    'Nutrien\n(NTR)':          'Nutrien',
    'Deere\n(DE)':             'Deere',
    'Goodyear\n(GT)':          'Goodyear',
    'Sherwin-Williams\n(SHW)': 'Sherwin-Williams',
    'Mattel\n(MAT)':           'Mattel',
    'Walmart\n(WMT)':          'Walmart',
    'Target\n(TGT)':           'Target',
}

ENTITY_ORDER = ['Nutrien', 'Deere', 'Goodyear', 'Sherwin-Williams',
                'Mattel', 'Walmart', 'Target']

GOOD_METRIC_KEYWORDS = [
    'reduction',  'recycled', 'renewable', 'reused', 'recycling rate',
    'diversion rate', 'women', 'sustainable packaging', 'deforestation-free',
    'biodiversity policy', 'tcfd', 'iso 14001', 'iso 9001', 'sbti',
    'cdp', 'validated', 'disclosure', 'certified', 'certifications',
    'training hours', 'training investment', 'living wage', 'minimum hourly',
    'human rights policy', 'due diligence process', 'audits conducted',
    'child labor policy', 'forced', 'modern slavery', 'conflict minerals',
    'freedom of association', 'indigenous peoples', 'community investment',
    'charitable', 'volunteer hours', 'local hiring', 'near-miss',
    'data privacy policy', 'anti-corruption', 'whistleblower', 'ethics hotline',
    'supplier code', 'suppliers audited', 'supplier diversity', 'responsible sourcing',
    'minorities', 'minority', 'ethnic', 'disabilities',
    'recycled input', 'renewable energy share',
]

RISK_METRIC_KEYWORDS = [
    'emission', 'intensity', 'consumption', 'withdrawal', 'discharge',
    'waste generated', 'hazardous', 'landfill', 'fines', 'penalties',
    'nox', 'sox', 'voc', 'particulate', 'plastic packaging',
    'turnover rate', 'pay gap', 'ceo-to-median', 'pay ratio',
    'incident rate', 'fatalities', 'serious injuries', 'osha violations',
    'non-conformances', 'grievances', 'recalls', 'safety incidents',
    'complaint rate', 'privacy breaches', 'ethics complaints',
    'ethics violations', 'terminated',
]


# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def is_good_metric(indicator_name):
    """Determine if higher value = lower risk for this metric."""
    name_lower = indicator_name.lower()
    for kw in GOOD_METRIC_KEYWORDS:
        if kw in name_lower:
            return True
    return False


def parse_value(val):
    """Parse raw ESG metric value from Excel into a float."""
    if pd.isna(val):
        return np.nan

    val_str = str(val).strip()

    if val_str.upper() in ('N/D', 'N/R', 'NAN', 'NONE', '-', ''):
        return np.nan

    val_lower = val_str.lower()
    if val_lower.startswith('yes'):
        return 1.0
    if val_lower.startswith('no'):
        return 0.0
    if val_lower.startswith('partial'):
        return 0.5

    grade_map = {
        'A': 0.95, 'A-': 0.85, 'B+': 0.75, 'B': 0.65, 'B-': 0.55,
        'C+': 0.45, 'C': 0.35, 'C-': 0.25, 'D+': 0.15, 'D': 0.10,
        'D-': 0.05, 'F': 0.0
    }
    if val_str in grade_map:
        return grade_map[val_str]

    clean = val_str.replace('~', '').replace(',', '').replace('$', '').strip()

    if '(' in clean:
        clean = clean[:clean.index('(')].strip()

    if ':1' in clean:
        try:
            return float(clean.replace(':1', ''))
        except ValueError:
            return np.nan

    if clean.endswith('%'):
        try:
            return float(clean[:-1]) / 100.0
        except ValueError:
            return np.nan

    try:
        return float(clean)
    except ValueError:
        return np.nan


def load_esg_data(filepath):
    """Load Environmental and Social data from Excel."""
    sheets = {
        'Environmental': 'Environmental Raw Data',
        'Social':        'Social Raw Data',
    }
    results = {}

    for dim_name, sheet_name in sheets.items():
        df_raw = pd.read_excel(filepath, sheet_name=sheet_name, header=None)

        company_cols = {}
        for col_idx in range(3, df_raw.shape[1]):
            header_val = str(df_raw.iloc[1, col_idx]).strip()
            for excel_key, internal_name in COMPANY_COLUMNS.items():
                if excel_key == header_val:
                    company_cols[col_idx] = internal_name
                    break

        records = []
        current_category = ''

        for row_idx in range(3, df_raw.shape[0]):
            cat = df_raw.iloc[row_idx, 0]
            if pd.notna(cat) and str(cat).strip():
                current_category = str(cat).strip()

            indicator = df_raw.iloc[row_idx, 1]
            if pd.isna(indicator) or not str(indicator).strip():
                continue
            indicator = str(indicator).strip()

            unit = str(df_raw.iloc[row_idx, 2]).strip() if pd.notna(df_raw.iloc[row_idx, 2]) else ''

            if unit.lower() == 'description':
                continue

            good = is_good_metric(indicator)

            for col_idx, company in company_cols.items():
                raw_val = df_raw.iloc[row_idx, col_idx]
                parsed = parse_value(raw_val)
                if not np.isnan(parsed):
                    records.append({
                        'company':   company,
                        'category':  current_category,
                        'indicator': indicator,
                        'unit':      unit,
                        'value':     parsed,
                        'is_good':   good,
                    })

        results[dim_name] = pd.DataFrame(records)

    return results


def normalize_and_label(df_dim, baseline_stats=None):
    """Z-score normalize each indicator across companies.

    Mode B (rolling baseline): baseline_stats=None
        Compute mean/std from current data. Used in Step 1 (FL training).
    Mode A (fixed baseline):   baseline_stats={indicator: {'mean', 'std', 'is_good'}}
        Use pre-computed mean/std from initial training. Used in Step 2 (risk scoring).

    Returns: (df_normalized, stats_dict)
    """
    records = []
    stats = {}

    if baseline_stats is None:
        # Mode B: rolling baseline — compute stats from current data
        for indicator, group in df_dim.groupby('indicator'):
            if len(group) < 3:
                continue
            values = group['value'].values
            mu = float(np.mean(values))
            sigma = float(np.std(values))
            if sigma < 1e-12:
                continue

            is_good = group['is_good'].iloc[0]
            stats[indicator] = {'mean': mu, 'std': sigma, 'is_good': is_good}

            for _, row in group.iterrows():
                z = (row['value'] - mu) / sigma
                risk_z = -z if is_good else z
                records.append({
                    'company':   row['company'],
                    'indicator': row['indicator'],
                    'category':  row['category'],
                    'risk_z':    risk_z,
                    'label':     1 if risk_z > 0 else 0,
                })
    else:
        # Mode A: fixed baseline — use pre-computed stats
        stats = baseline_stats
        for indicator, group in df_dim.groupby('indicator'):
            if indicator not in baseline_stats:
                continue
            bs = baseline_stats[indicator]
            mu = bs['mean']
            sigma = bs['std']
            is_good = bs['is_good']

            for _, row in group.iterrows():
                z = (row['value'] - mu) / sigma
                risk_z = -z if is_good else z
                records.append({
                    'company':   row['company'],
                    'indicator': row['indicator'],
                    'category':  row['category'],
                    'risk_z':    risk_z,
                    'label':     1 if risk_z > 0 else 0,
                })

    return pd.DataFrame(records), stats


def build_features(df_norm, categories=None):
    """Build per-company feature matrices for logistic regression.

    If categories is provided (from Step 1), use that fixed list to ensure
    feature vector alignment with the global model.
    """
    if categories is None:
        categories = sorted(df_norm['category'].unique())
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    n_cats = len(categories)

    company_data = {}
    for company, grp in df_norm.groupby('company'):
        X_rows, y_rows = [], []
        for _, row in grp.iterrows():
            cat = row['category']
            if cat not in cat_to_idx:
                continue
            feat = np.zeros(1 + n_cats)
            feat[0] = row['risk_z']
            feat[1 + cat_to_idx[cat]] = 1.0
            X_rows.append(feat)
            y_rows.append(row['label'])
        if X_rows:
            company_data[company] = (np.array(X_rows), np.array(y_rows))

    return company_data, categories


# =============================================================================
# FEDERATED LEARNING — CLIENT & SERVER
# =============================================================================

class FLClient:
    """A participating firm in the FL system."""

    def __init__(self, name, tier, role, n_features):
        self.name = name
        self.tier = tier
        self.role = role
        self.n_features = n_features
        self.model = None
        self.local_params = None

    def train_local(self, X, y, global_params=None):
        """Train logistic regression locally."""
        self.model = LogisticRegression(
            max_iter=300, C=1.0, solver='lbfgs', random_state=42,
        )
        y_train = y.copy()
        if len(np.unique(y_train)) < 2:
            y_train[0] = 1 - y_train[0]

        if global_params is not None:
            self.model.fit(X, y_train)
            self.model.coef_ = global_params['coef'].copy()
            self.model.intercept_ = global_params['intercept'].copy()
            self.model.warm_start = True
            self.model.fit(X, y_train)
        else:
            self.model.fit(X, y_train)

        self.local_params = {
            'coef':      self.model.coef_.copy(),
            'intercept': self.model.intercept_.copy(),
            'n_samples': len(y),
        }
        return self.local_params

    def hash_params(self):
        """SHA-256 hash of model parameters (privacy simulation)."""
        raw = np.concatenate([
            self.local_params['coef'].flatten(),
            self.local_params['intercept'].flatten(),
        ]).tobytes()
        return hashlib.sha256(raw).hexdigest()

    def receive_global(self, global_params):
        """Receive aggregated global model from server."""
        if self.model is not None:
            self.model.coef_ = global_params['coef'].copy()
            self.model.intercept_ = global_params['intercept'].copy()

    def set_global_model(self, global_params, n_features):
        """Set model to global parameters without local training (Step 2)."""
        self.model = LogisticRegression(
            max_iter=300, C=1.0, solver='lbfgs', random_state=42,
        )
        # Fit on dummy data to initialise sklearn internals
        dummy_X = np.zeros((2, n_features))
        dummy_y = np.array([0, 1])
        self.model.fit(dummy_X, dummy_y)
        # Override with global params
        self.model.coef_ = global_params['coef'].copy()
        self.model.intercept_ = global_params['intercept'].copy()
        self.local_params = {
            'coef': global_params['coef'].copy(),
            'intercept': global_params['intercept'].copy(),
            'n_samples': 0,
        }

    def predict_risk_score(self, X):
        """Average P(high_risk) across all local metrics -> R_i^local."""
        if self.model is None:
            return 0.5
        return float(np.mean(self.model.predict_proba(X)[:, 1]))


class FLServer:
    """Central coordination server for FedAvg aggregation."""

    def __init__(self):
        self.global_params = None

    def aggregate(self, client_params_list):
        """FedAvg: weighted average of client parameters by sample count."""
        total_n = sum(p['n_samples'] for p in client_params_list)
        avg_coef = np.zeros_like(client_params_list[0]['coef'])
        avg_intercept = np.zeros_like(client_params_list[0]['intercept'])

        for p in client_params_list:
            w = p['n_samples'] / total_n
            avg_coef += w * p['coef']
            avg_intercept += w * p['intercept']

        self.global_params = {'coef': avg_coef, 'intercept': avg_intercept}
        return self.global_params


# =============================================================================
# RISK PROPAGATION
# =============================================================================

def propagate_risk(local_risks):
    """Compute R_i^total using upstream risk propagation.
    R_i^total = Sum_{j in S_i} alpha_ij * R_j^total + beta_i * R_i^local
    Processed in tier order: Tier 2 -> Tier 1 -> Focal -> Downstream.

    Note: risk flows upstream -> downstream only.
    Downstream entities (Walmart, Target) incorporate Mattel's R_total,
    but Mattel does NOT incorporate downstream R_total.
    """
    total = {}
    for tier in ('Tier 2', 'Tier 1', 'Focal', 'Downstream'):
        for entity, cfg in SUPPLY_CHAIN.items():
            if cfg['tier'] != tier:
                continue
            w = PROPAGATION_WEIGHTS[entity]
            upstream_sum = sum(
                w['alpha'][s] * total.get(s, local_risks.get(s, 0.5))
                for s in w['alpha']
            )
            total[entity] = upstream_sum + w['beta'] * local_risks.get(entity, 0.5)
    return total


# =============================================================================
# CONSOLE REPORTING
# =============================================================================

def print_header(title, char='=', width=80):
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def print_bar(value, length=30):
    filled = int(value * length)
    return '#' * filled + '.' * (length - filled)


# =============================================================================
# STEP 1: FL TRAINING (Low-Frequency)
# =============================================================================

def run_fl_training(filepath, scenario_label, fl_rounds=FL_ROUNDS, verbose=True):
    """Step 1: Federated Learning model training.

    Uses rolling baseline (Plan B) for normalization.
    Returns dict with global_params, baseline_stats, categories per dimension.
    """
    if verbose:
        print_header(f"STEP 1 — FL TRAINING: {scenario_label}")
        print(f"  Data file: {os.path.basename(filepath)}")
        print(f"  Normalization: Rolling baseline (Plan B)")
        print(f"  Rounds: {fl_rounds}")

    raw_data = load_esg_data(filepath)

    if verbose:
        print_header("DATA LOADING", char='-')
        for dim, df in raw_data.items():
            print(f"  {dim:15s}: {df['indicator'].nunique():3d} indicators, "
                  f"{df['company'].nunique()} companies, {len(df)} data points")

    fl_result = {}

    for dim_name, df_raw in raw_data.items():
        # Normalize with rolling baseline (Mode B)
        df_norm, baseline_stats = normalize_and_label(df_raw, baseline_stats=None)
        company_data, categories = build_features(df_norm)

        if verbose:
            print(f"\n  {dim_name} — sub-categories: {categories}")
            for company in ENTITY_ORDER:
                if company in company_data:
                    X, y = company_data[company]
                    print(f"    {company:20s}: {len(y):3d} samples  "
                          f"(high-risk: {int(y.sum()):2d}, low-risk: {int(len(y) - y.sum()):2d})")

        # FL training loop
        if verbose:
            print_header(f"FL ROUNDS — {dim_name.upper()}", char='-')

        n_features = list(company_data.values())[0][0].shape[1]

        clients = {}
        for entity in ENTITY_ORDER:
            if entity in company_data:
                cfg = SUPPLY_CHAIN[entity]
                clients[entity] = FLClient(entity, cfg['tier'], cfg['role'], n_features)

        server = FLServer()
        global_params = None
        prev_coef_norm = None

        for rnd in range(1, fl_rounds + 1):
            if verbose:
                print(f"\n  -- Round {rnd:2d}/{fl_rounds} --")
            round_params = []

            for entity, client in clients.items():
                X, y = company_data[entity]
                params = client.train_local(X, y, global_params)
                h = client.hash_params()
                round_params.append(params)
                if verbose:
                    print(f"    ^ {entity:20s} ({client.tier:12s}) -> Server  "
                          f"| hash: {h[:20]}...  | n={params['n_samples']}")

            global_params = server.aggregate(round_params)
            coef_norm = float(np.linalg.norm(global_params['coef']))
            delta = abs(coef_norm - prev_coef_norm) if prev_coef_norm is not None else float('nan')
            prev_coef_norm = coef_norm

            if verbose:
                print(f"    [+] Server FedAvg complete  | ||coef||={coef_norm:.6f}  "
                      f"| delta={delta:.6f}")

            for entity, client in clients.items():
                client.receive_global(global_params)

            if verbose:
                print(f"    v Global model -> {len(clients)} clients")

        # Parameter transmission chain
        if verbose:
            print(f"\n  {'-' * 66}")
            print(f"  PARAMETER TRANSMISSION CHAIN — {dim_name.upper()}")
            print(f"  {'-' * 66}")

            for t2 in ('Nutrien', 'Deere'):
                if t2 in clients and 'Goodyear' in clients:
                    h = clients[t2].hash_params()
                    print(f"    {t2:20s} (Tier 2) --> {'Goodyear':20s} (Tier 1)  | {h[:20]}...")

            for t1 in ('Goodyear', 'Sherwin-Williams'):
                if t1 in clients and 'Mattel' in clients:
                    h = clients[t1].hash_params()
                    print(f"    {t1:20s} (Tier 1) --> {'Mattel':20s} (Focal)  | {h[:20]}...")

            if 'Mattel' in clients:
                h = clients['Mattel'].hash_params()
                for ds in ('Walmart', 'Target'):
                    if ds in clients:
                        print(f"    {'Mattel':20s} (Focal)  --> {ds:20s} (Down.)  | {h[:20]}...")

        fl_result[dim_name] = {
            'global_params':  deepcopy(global_params),
            'baseline_stats': baseline_stats,
            'categories':     categories,
            'n_features':     n_features,
        }

    if verbose:
        print_header("FL TRAINING COMPLETE", char='-')
        for dim_name in fl_result:
            gp = fl_result[dim_name]['global_params']
            print(f"  {dim_name}: ||coef|| = {np.linalg.norm(gp['coef']):.6f}, "
                  f"baseline indicators = {len(fl_result[dim_name]['baseline_stats'])}")

    return fl_result


# =============================================================================
# STEP 2: RISK SCORING (High-Frequency)
# =============================================================================

def run_risk_scoring(filepath, scenario_label, fl_result, verbose=True):
    """Step 2: Risk scoring using pre-trained global model + fixed baseline.

    Uses fixed baseline (Plan A) from fl_result for normalization.
    No local training — only inference with the global model.
    Returns: {dim: {'local_risks': {...}, 'total_risks': {...}}}
    """
    if verbose:
        print_header(f"STEP 2 — RISK SCORING: {scenario_label}")
        print(f"  Data file: {os.path.basename(filepath)}")
        print(f"  Normalization: Fixed baseline (Plan A)")
        print(f"  Model: Global model from Step 1 (no re-training)")

    raw_data = load_esg_data(filepath)

    scoring_results = {}

    for dim_name, df_raw in raw_data.items():
        dim_fl = fl_result[dim_name]
        baseline_stats = dim_fl['baseline_stats']
        global_params = dim_fl['global_params']
        categories = dim_fl['categories']
        n_features = dim_fl['n_features']

        # Normalize with fixed baseline (Mode A)
        df_norm, _ = normalize_and_label(df_raw, baseline_stats=baseline_stats)
        company_data, _ = build_features(df_norm, categories=categories)

        if verbose:
            print(f"\n  {dim_name} — scoring {len(company_data)} entities")

        # Create clients with global model (no local training)
        clients = {}
        for entity in ENTITY_ORDER:
            if entity in company_data:
                cfg = SUPPLY_CHAIN[entity]
                client = FLClient(entity, cfg['tier'], cfg['role'], n_features)
                client.set_global_model(global_params, n_features)
                clients[entity] = client

        # Compute local risk scores
        local_risks = {}
        if verbose:
            print(f"\n  {'Entity':20s}  {'Tier':12s}  {'R_local':>8s}  Bar")
            print(f"  {'-' * 66}")
        for entity in ENTITY_ORDER:
            if entity in clients:
                X, y = company_data[entity]
                score = clients[entity].predict_risk_score(X)
                local_risks[entity] = score
                if verbose:
                    tier = SUPPLY_CHAIN[entity]['tier']
                    bar = print_bar(score)
                    print(f"  {entity:20s}  {tier:12s}  {score:.4f}  |{bar}|")

        # Propagate risk (upstream -> downstream)
        total_risks = propagate_risk(local_risks)

        if verbose:
            print(f"\n  PROPAGATED RISK — {dim_name.upper()}")
            print(f"  {'Entity':20s}  {'Tier':12s}  {'R_local':>8s}  {'R_total':>8s}  Decomposition")
            print(f"  {'-' * 78}")
            for entity in ENTITY_ORDER:
                if entity not in total_risks:
                    continue
                r_tot = total_risks[entity]
                r_loc = local_risks.get(entity, 0.5)
                w = PROPAGATION_WEIGHTS[entity]
                tier = SUPPLY_CHAIN[entity]['tier']

                parts = [f"B*{r_loc:.4f}={w['beta'] * r_loc:.4f}"]
                for s, a in w['alpha'].items():
                    r_s = total_risks.get(s, 0.5)
                    parts.append(f"a({s})*{r_s:.4f}={a * r_s:.4f}")

                print(f"  {entity:20s}  {tier:12s}  {r_loc:.4f}  {r_tot:.4f}  "
                      f"= {' + '.join(parts)}")

        scoring_results[dim_name] = {
            'local_risks': local_risks,
            'total_risks': total_risks,
        }

    # Composite ESG risk
    if verbose:
        print(f"\n  COMPOSITE ESG RISK (50% E + 50% S)")
        print(f"  {'Entity':20s}  {'Tier':12s}  {'E_total':>8s}  {'S_total':>8s}  {'Composite':>10s}")
        print(f"  {'-' * 70}")
        for entity in ENTITY_ORDER:
            tier = SUPPLY_CHAIN[entity]['tier']
            e = scoring_results.get('Environmental', {}).get('total_risks', {}).get(entity, 0.5)
            s = scoring_results.get('Social', {}).get('total_risks', {}).get(entity, 0.5)
            composite = 0.5 * e + 0.5 * s
            bar = print_bar(composite)
            print(f"  {entity:20s}  {tier:12s}  {e:.4f}  {s:.4f}  {composite:10.4f}  |{bar}|")

    return scoring_results


# =============================================================================
# COMPARISON REPORTING
# =============================================================================

def print_comparison(title, results_a, results_b, label_a, label_b):
    """Compare risk scores between two Step 2 scoring runs."""
    print_header(f"COMPARISON: {title}")

    for dim in ('Environmental', 'Social'):
        a_local = results_a.get(dim, {}).get('local_risks', {})
        a_total = results_a.get(dim, {}).get('total_risks', {})
        b_local = results_b.get(dim, {}).get('local_risks', {})
        b_total = results_b.get(dim, {}).get('total_risks', {})

        print(f"\n  {dim.upper()} — Local Risk (R_local)")
        print(f"  {'Entity':20s}  {'Tier':12s}  {label_a:>10s}  {label_b:>10s}  {'Delta':>10s}  {'Change%':>10s}")
        print(f"  {'-' * 78}")
        for entity in ENTITY_ORDER:
            tier = SUPPLY_CHAIN[entity]['tier']
            va = a_local.get(entity, 0.5)
            vb = b_local.get(entity, 0.5)
            delta = vb - va
            pct = (delta / va * 100) if va != 0 else 0.0
            marker = " ***" if abs(pct) > 1.0 else ""
            print(f"  {entity:20s}  {tier:12s}  {va:10.4f}  {vb:10.4f}  {delta:+10.4f}  {pct:+9.2f}%{marker}")

        print(f"\n  {dim.upper()} — Propagated Risk (R_total)")
        print(f"  {'Entity':20s}  {'Tier':12s}  {label_a:>10s}  {label_b:>10s}  {'Delta':>10s}  {'Change%':>10s}")
        print(f"  {'-' * 78}")
        for entity in ENTITY_ORDER:
            tier = SUPPLY_CHAIN[entity]['tier']
            va = a_total.get(entity, 0.5)
            vb = b_total.get(entity, 0.5)
            delta = vb - va
            pct = (delta / va * 100) if va != 0 else 0.0
            marker = " ***" if abs(pct) > 1.0 else ""
            print(f"  {entity:20s}  {tier:12s}  {va:10.4f}  {vb:10.4f}  {delta:+10.4f}  {pct:+9.2f}%{marker}")

    # Focal firm summary
    print(f"\n  FOCAL FIRM (MATTEL) IMPACT SUMMARY")
    print(f"  {'-' * 60}")
    for dim in ('Environmental', 'Social'):
        a_tot = results_a.get(dim, {}).get('total_risks', {}).get('Mattel', 0.5)
        b_tot = results_b.get(dim, {}).get('total_risks', {}).get('Mattel', 0.5)
        delta = b_tot - a_tot
        pct = (delta / a_tot * 100) if a_tot != 0 else 0.0
        print(f"  {dim:15s}: {label_a}={a_tot:.4f} -> {label_b}={b_tot:.4f}  "
              f"(delta={delta:+.4f}, {pct:+.2f}%)")

    # Propagation chain trace (for each dimension with non-zero change)
    print(f"\n  RISK PROPAGATION CHAIN TRACE")
    print(f"  {'-' * 60}")
    chain = ['Nutrien', 'Deere', 'Goodyear', 'Sherwin-Williams', 'Mattel', 'Walmart', 'Target']
    for dim in ('Environmental', 'Social'):
        a_local = results_a.get(dim, {}).get('local_risks', {})
        b_local = results_b.get(dim, {}).get('local_risks', {})
        a_total = results_a.get(dim, {}).get('total_risks', {})
        b_total = results_b.get(dim, {}).get('total_risks', {})

        # Find entities with local risk changes
        changed = [e for e in chain
                   if abs(b_local.get(e, 0.5) - a_local.get(e, 0.5)) > 1e-6]
        if not changed:
            continue

        print(f"\n  {dim}:")
        for src in changed:
            src_delta = b_local[src] - a_local[src]
            print(f"    {src} R_local: {a_local[src]:.4f} -> {b_local[src]:.4f} "
                  f"(delta={src_delta:+.4f}) [ORIGIN]")

        # Trace propagation through tiers
        for entity in chain:
            delta_total = b_total.get(entity, 0.5) - a_total.get(entity, 0.5)
            delta_local = b_local.get(entity, 0.5) - a_local.get(entity, 0.5)
            propagated = delta_total - delta_local
            if abs(propagated) > 1e-7:
                w = PROPAGATION_WEIGHTS[entity]
                parts = []
                for s, a in w['alpha'].items():
                    ds = b_total.get(s, 0.5) - a_total.get(s, 0.5)
                    if abs(ds) > 1e-7:
                        parts.append(f"a({s})={a:.2f} x delta({s})={ds:+.4f} = {a*ds:+.6f}")
                if parts:
                    print(f"    {entity} R_total: {a_total[entity]:.4f} -> {b_total[entity]:.4f} "
                          f"(delta={delta_total:+.4f})")
                    for p in parts:
                        print(f"      <- {p}")


# =============================================================================
# MAIN — TWO-STEP DEMO
# =============================================================================

def main():
    print_header("FEDERATED LEARNING FOR ESG RISK ASSESSMENT IN SUPPLY CHAINS")
    print("  Two-Step Architecture Demonstration")
    print("  Step 1 (Low-Freq): FL Training — FedAvg, rolling baseline")
    print("  Step 2 (High-Freq): Risk Scoring — fixed baseline, no re-training")

    # Supply chain structure
    print_header("SUPPLY CHAIN STRUCTURE")
    for entity in ENTITY_ORDER:
        cfg = SUPPLY_CHAIN[entity]
        up = ', '.join(cfg['upstream']) if cfg['upstream'] else 'None'
        print(f"  {cfg['tier']:12s} | {entity:20s} | {cfg['role']:25s} | Upstream: {up}")

    # Weight verification
    print_header("RISK PROPAGATION WEIGHTS")
    for entity in ENTITY_ORDER:
        w = PROPAGATION_WEIGHTS[entity]
        parts = [f"a({s})={a:.2f}" for s, a in w['alpha'].items()]
        alpha_str = ', '.join(parts) if parts else 'N/A'
        total = w['beta'] + sum(w['alpha'].values())
        print(f"  {entity:20s} | B={w['beta']:.2f} | {alpha_str:40s} | Sum={total:.2f}")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    path_m0 = os.path.join(script_dir, 'ESG_Raw_Data_Supply_Chain_M0.xlsx')
    path_m1 = os.path.join(script_dir, 'ESG_Raw_Data_Supply_Chain_M1.xlsx')
    path_m2 = os.path.join(script_dir, 'ESG_Raw_Data_Supply_Chain_M2.xlsx')
    path_m3 = os.path.join(script_dir, 'ESG_Raw_Data_Supply_Chain_M3.xlsx')

    # =================================================================
    # STEP 1: Initial FL Training on M0
    # =================================================================
    fl_result = run_fl_training(path_m0, "Initial Training on M0")

    # =================================================================
    # STEP 2a: Baseline risk snapshot (M0)
    # =================================================================
    scores_m0 = run_risk_scoring(path_m0, "M0 — Baseline Snapshot", fl_result)

    # =================================================================
    # STEP 2b: M1 — Nutrien GHG Emissions Doubled (E1-E4)
    # =================================================================
    scores_m1 = run_risk_scoring(path_m1, "M1 — Nutrien GHG Doubled", fl_result)
    print_comparison(
        "M0 vs M1: Nutrien (Tier 2) GHG Emissions Doubled",
        scores_m0, scores_m1, "M0", "M1",
    )

    # =================================================================
    # STEP 2c: M2 — Deere Workforce Diversity Improved
    # =================================================================
    if os.path.exists(path_m2):
        scores_m2 = run_risk_scoring(path_m2, "M2 — Deere Diversity Improved", fl_result)
        print_comparison(
            "M1 vs M2: Deere (Tier 2) Workforce Diversity Improved",
            scores_m1, scores_m2, "M1", "M2",
        )
    else:
        print(f"\n  [SKIP] M2 file not found: {os.path.basename(path_m2)}")
        scores_m2 = None

    # =================================================================
    # STEP 2d: M3 — Walmart GHG Improved (Downstream)
    # =================================================================
    if os.path.exists(path_m3):
        scores_m3 = run_risk_scoring(path_m3, "M3 — Walmart GHG Improved", fl_result)

        # Compare M2 vs M3 to show downstream doesn't affect focal
        prev_scores = scores_m2 if scores_m2 is not None else scores_m1
        prev_label = "M2" if scores_m2 is not None else "M1"
        print_comparison(
            f"{prev_label} vs M3: Walmart (Downstream) GHG Improved",
            prev_scores, scores_m3, prev_label, "M3",
        )

        print_header("KEY OBSERVATION: DOWNSTREAM DOES NOT AFFECT FOCAL")
        mattel_prev_e = prev_scores.get('Environmental', {}).get('total_risks', {}).get('Mattel', 0.5)
        mattel_m3_e = scores_m3.get('Environmental', {}).get('total_risks', {}).get('Mattel', 0.5)
        walmart_prev_e = prev_scores.get('Environmental', {}).get('total_risks', {}).get('Walmart', 0.5)
        walmart_m3_e = scores_m3.get('Environmental', {}).get('total_risks', {}).get('Walmart', 0.5)
        print(f"  Walmart Environmental R_total: {walmart_prev_e:.4f} -> {walmart_m3_e:.4f}  "
              f"(delta={walmart_m3_e - walmart_prev_e:+.4f})")
        print(f"  Mattel  Environmental R_total: {mattel_prev_e:.4f} -> {mattel_m3_e:.4f}  "
              f"(delta={mattel_m3_e - mattel_prev_e:+.4f})")
        print(f"  -> Mattel's risk is UNAFFECTED by downstream improvement.")
        print(f"     Mattel can OBSERVE Walmart's score change (monitoring),")
        print(f"     but the change does not propagate upstream.")
    else:
        print(f"\n  [SKIP] M3 file not found: {os.path.basename(path_m3)}")
        scores_m3 = None

    # =================================================================
    # STEP 1 AGAIN: FL Re-Training on M3 (demonstrate model update)
    # =================================================================
    if scores_m3 is not None:
        print_header("FL MODEL UPDATE DEMONSTRATION", char='*')
        print("  After collecting M0-M3 risk data, the system performs a new FL round.")
        print("  This re-trains the global model with the latest data (M3).")
        print("  The new model captures updated ESG risk patterns.")

        fl_result_m3 = run_fl_training(path_m3, "Re-Training on M3 Data")

        # Score M3 with the NEW model
        scores_m3_new = run_risk_scoring(
            path_m3, "M3 — With Updated Global Model", fl_result_m3,
        )

        print_comparison(
            "M3 Old Model vs M3 New Model (FL Re-Training Effect)",
            scores_m3, scores_m3_new, "OldModel", "NewModel",
        )

        print_header("FL RE-TRAINING SIGNIFICANCE")
        print("  The old model was trained on M0 baseline data.")
        print("  The new model was re-trained on M3 data (incorporating all changes).")
        print("  Differences in risk scores reflect the model's updated understanding")
        print("  of 'what ESG indicator patterns predict risk'.")
        print("  This demonstrates the value of periodic FL updates (Step 1)")
        print("  alongside continuous risk monitoring (Step 2).")

    # =================================================================
    # SYSTEM SUMMARY
    # =================================================================
    print_header("TWO-STEP FL SYSTEM SUMMARY")
    print(f"  {'='*60}")
    print(f"  STEP 1 — FL Training (Low-Frequency)")
    print(f"  {'='*60}")
    print(f"  Algorithm:        FedAvg (Federated Averaging)")
    print(f"  Rounds:           {FL_ROUNDS}")
    print(f"  Normalization:    Rolling baseline (Plan B)")
    print(f"  Purpose:          Learn risk classification rules")
    print(f"  Output:           Global model + baseline statistics")
    print()
    print(f"  {'='*60}")
    print(f"  STEP 2 — Risk Scoring (High-Frequency)")
    print(f"  {'='*60}")
    print(f"  Model:            Pre-trained global model (no re-training)")
    print(f"  Normalization:    Fixed baseline from Step 1 (Plan A)")
    print(f"  Propagation:      Upstream -> Downstream (one-directional)")
    print(f"  Purpose:          Monitor ESG risk changes in real-time")
    print(f"  Key property:     Downstream changes visible but do not")
    print(f"                    affect upstream/focal risk scores")
    print()
    print(f"  {'='*60}")
    print(f"  DEMO SCENARIOS")
    print(f"  {'='*60}")
    print(f"  M0: Baseline ESG data")
    print(f"  M1: Nutrien (Tier 2) GHG emissions doubled -> upstream risk up")
    print(f"  M2: Deere (Tier 2) workforce diversity improved -> social risk down")
    print(f"  M3: Walmart (Downstream) GHG improved -> downstream only, focal unchanged")
    print(f"  FL Re-train: Model updated on M3 -> demonstrates system design value")
    print()
    print(f"  +-------------------------------------------------------------------+")
    print(f"  |  KEY PRIVACY GUARANTEE                                            |")
    print(f"  |  Raw ESG data never leaves the participating firm.                |")
    print(f"  |  Only model parameters (Step 1) and encrypted risk scores         |")
    print(f"  |  (Step 2) are transmitted through the supply chain.               |")
    print(f"  +-------------------------------------------------------------------+")
    print("=" * 80)


# ============ DASHBOARD APPENDED BELOW ============

# =============================================================================
# =============================================================================
# DASHBOARD — Two-Step FL ESG Risk Monitor
# =============================================================================
# =============================================================================
#
# Run with:  streamlit run FL_TwoStep_Dashboard.py
# CLI mode:  python  FL_TwoStep_Dashboard.py
#
# Architecture:
#   Step 1 (Low-Freq):  run_fl_training()  — FedAvg on M0, rolling baseline
#   Step 2 (High-Freq): run_risk_scoring() — fixed baseline, no re-training
#
# Four demo scenarios (M0 → M1 → M2 → M3) + M3 re-training:
#   M0: Baseline ESG data
#   M1: Nutrien GHG emissions doubled (upstream E risk propagates down)
#   M2: Deere workforce diversity improved (S risk isolated, E unchanged)
#   M3: Walmart GHG improved (downstream only, focal firm unaffected)
#   M3-Retrain: New FL round on M3 data — model "cognition" updated
#
# Visibility rules (federated privacy model):
#   Nutrien          → can see: Nutrien, Goodyear
#   Deere            → can see: Deere, Goodyear
#   Goodyear         → can see: Nutrien, Deere, Goodyear, Mattel
#   Sherwin-Williams → can see: Sherwin-Williams, Mattel
#   Mattel           → can see: Goodyear, Sherwin-Williams, Mattel, Walmart, Target
#   Walmart          → can see: Mattel, Walmart
#   Target           → can see: Mattel, Target
# =============================================================================

import time
from datetime import datetime, timedelta
from copy import deepcopy

try:
    import streamlit as st
    import plotly.graph_objects as go
    _HAVE_DASHBOARD = True
except ImportError:
    _HAVE_DASHBOARD = False


# ---- Supply chain visibility map ----------------------------------------
SUPPLY_CHAIN_VIS = {
    'Nutrien': {
        'can_see':          ['Nutrien', 'Goodyear'],
        'direct_suppliers': [],
        'direct_customers': ['Goodyear'],
    },
    'Deere': {
        'can_see':          ['Deere', 'Goodyear'],
        'direct_suppliers': [],
        'direct_customers': ['Goodyear'],
    },
    'Goodyear': {
        'can_see':          ['Nutrien', 'Deere', 'Goodyear', 'Mattel'],
        'direct_suppliers': ['Nutrien', 'Deere'],
        'direct_customers': ['Mattel'],
    },
    'Sherwin-Williams': {
        'can_see':          ['Sherwin-Williams', 'Mattel'],
        'direct_suppliers': [],
        'direct_customers': ['Mattel'],
    },
    'Mattel': {
        'can_see':          ['Goodyear', 'Sherwin-Williams', 'Mattel', 'Walmart', 'Target'],
        'direct_suppliers': ['Goodyear', 'Sherwin-Williams'],
        'direct_customers': ['Walmart', 'Target'],
    },
    'Walmart': {
        'can_see':          ['Mattel', 'Walmart'],
        'direct_suppliers': ['Mattel'],
        'direct_customers': [],
    },
    'Target': {
        'can_see':          ['Mattel', 'Target'],
        'direct_suppliers': ['Mattel'],
        'direct_customers': [],
    },
}

NODE_POS = {
    'Nutrien':          (0.0, 1.6),
    'Deere':            (0.0, 0.4),
    'Goodyear':         (1.6, 1.6),
    'Sherwin-Williams': (1.6, 0.4),
    'Mattel':           (3.2, 1.0),
    'Walmart':          (4.8, 1.6),
    'Target':           (4.8, 0.4),
}

SC_EDGES = [
    ('Nutrien',   'Goodyear'),
    ('Deere',     'Goodyear'),
    ('Goodyear',  'Mattel'),
    ('Sherwin-Williams', 'Mattel'),
    ('Mattel',    'Walmart'),
    ('Mattel',    'Target'),
]

DISPLAY_NAMES = {
    'Goodyear': 'Goodyear Tire & Rubber',
}

# Scenario definitions for the Two-Step system
SCENARIO_OPTIONS = [
    'M0 — Baseline',
    'M1 — Nutrien GHG ×2 (Upstream E Risk)',
    'M2 — Deere Diversity Improved (S Risk)',
    'M3 — Walmart GHG Improved (Downstream)',
    'M3 — Retrained Model (New FL Round)',
]
SCENARIO_KEY_MAP = {
    'M0 — Baseline':                        'M0',
    'M1 — Nutrien GHG ×2 (Upstream E Risk)': 'M1',
    'M2 — Deere Diversity Improved (S Risk)': 'M2',
    'M3 — Walmart GHG Improved (Downstream)': 'M3',
    'M3 — Retrained Model (New FL Round)':    'M3-Retrained',
}
SCENARIO_DESC = {
    'M0': 'Baseline ESG data. Global model trained on M0 (Step 1). Fixed baseline established.',
    'M1': 'Nutrien GHG emissions doubled (E1-E4). Risk propagates: Nutrien → Goodyear → Mattel.',
    'M2': 'Deere workforce diversity improved. Social risk drops; Environmental is isolated and unchanged.',
    'M3': 'Walmart GHG improved (downstream only). Mattel\'s risk is unaffected — one-directional flow verified.',
    'M3-Retrained': 'New FL round trained on M3 data. Global model "cognition" updated — same data, different risk scores.',
}

DASH_CSS = """
<style>
  .dash-header {
    font-size: 2.0rem; color: #1e3a8a; font-weight: 800; margin-bottom: 0.4rem;
  }
  .dash-sub {
    font-size: 1.05rem; color: #64748b; margin-bottom: 1rem;
  }
  .card-blue   { background: linear-gradient(135deg,#667eea,#764ba2);
                  padding:0.9rem; border-radius:12px; color:white; }
  .card-red    { background: linear-gradient(135deg,#ff6b6b,#ee5a6f);
                  padding:0.9rem; border-radius:12px; color:white; }
  .card-green  { background: linear-gradient(135deg,#48dbfb,#0abde3);
                  padding:0.9rem; border-radius:12px; color:white; }
  .card-orange { background: linear-gradient(135deg,#f6c768,#f59e0b);
                  padding:0.9rem; border-radius:12px; color:white; }
  .card-live   { background: linear-gradient(135deg,#f87171,#dc2626);
                  padding:0.9rem; border-radius:12px; color:white;
                  border: 2px solid #7f1d1d; }
  .formula-box {
    background:#f8fafc; border-left:4px solid #2563eb;
    padding:0.7rem 1rem; border-radius:6px;
    font-family:monospace; font-size:0.85rem; line-height:1.6;
  }
  .conf-node {
    background:#f1f5f9; border:2px dashed #94a3b8;
    border-radius:8px; padding:0.5rem 0.8rem;
    display:inline-block; font-size:0.82rem; color:#64748b;
  }
  .tiny { font-size:0.80rem; color:#475569; }
  .weight-sum-ok  { color:#16a34a; font-weight:600; }
  .weight-sum-bad { color:#dc2626; font-weight:600; }
  .live-banner {
    background: linear-gradient(135deg, #fef3c7, #fde68a);
    border: 2px solid #f59e0b;
    border-radius: 10px; padding: 0.7rem 1rem;
    color: #78350f; font-weight: 600; margin-bottom: 0.8rem;
  }
  .privacy-badge {
    background: #f1f5f9; border: 1.5px solid #94a3b8;
    border-radius: 6px; padding: 0.35rem 0.7rem;
    color: #475569; font-size: 0.82rem; display: inline-block;
  }
  .step1-badge {
    background: linear-gradient(135deg,#3b82f6,#1d4ed8);
    color:white; border-radius:8px; padding:0.5rem 0.9rem;
    font-size:0.85rem; font-weight:700; display:inline-block;
  }
  .step2-badge {
    background: linear-gradient(135deg,#10b981,#065f46);
    color:white; border-radius:8px; padding:0.5rem 0.9rem;
    font-size:0.85rem; font-weight:700; display:inline-block;
  }
  .scenario-box {
    background:#f0f9ff; border:1.5px solid #7dd3fc;
    border-radius:8px; padding:0.6rem 1rem;
    font-size:0.85rem; color:#0c4a6e; margin-bottom:0.6rem;
  }
</style>
"""


# =========================================================================
# Helper: propagate risk with CUSTOM weight dict
# =========================================================================

def propagate_risk_custom(local_risks, pw):
    total = {}
    for tier in ('Tier 2', 'Tier 1', 'Focal', 'Downstream'):
        for entity, cfg in SUPPLY_CHAIN.items():
            if cfg['tier'] != tier:
                continue
            w = pw[entity]
            upstream_sum = sum(
                w['alpha'][s] * total.get(s, local_risks.get(s, 0.5))
                for s in w['alpha']
            )
            total[entity] = upstream_sum + w['beta'] * local_risks.get(entity, 0.5)
    return total


# =========================================================================
# Visualization helpers
# =========================================================================

def _risk_level(score):
    if score >= 0.70:
        return 'High Risk',   '#dc2626'
    elif score >= 0.40:
        return 'Medium Risk', '#d97706'
    return 'Low Risk',        '#16a34a'


def _card_class(score):
    if score >= 0.70:
        return 'card-red'
    elif score >= 0.40:
        return 'card-orange'
    return 'card-green'


def make_gauge(value, title, height=250, threshold=0.70):
    n = 30
    steps = []
    for i in range(n):
        frac, fn = i / n, (i + 1) / n
        if frac < 0.5:
            t = frac * 2
            r, g, b = int(t * 255), 200, 0
        else:
            t = (frac - 0.5) * 2
            r, g, b = 255, int(200 - t * 200), 0
        steps.append({'range': [frac, fn], 'color': f'rgb({r},{g},{b})'})

    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=value,
        number={'font': {'size': 30, 'color': '#1e3a8a'}, 'valueformat': '.4f'},
        title={'text': title, 'font': {'size': 13, 'color': '#334155'}},
        domain={'x': [0, 1], 'y': [0.18, 1.0]},
        gauge={
            'axis': {
                'range': [0, 1],
                'tickvals': [0.0, 0.4, 0.7, 1.0],
                'ticktext': ['0', '0.4', '0.7', '1'],
                'tickfont': {'size': 10},
            },
            'bar': {'color': '#1e3a8a', 'thickness': 0.16},
            'steps': steps,
            'threshold': {
                'line': {'color': '#7c3aed', 'width': 3},
                'thickness': 0.80,
                'value': threshold,
            },
            'bgcolor': 'white',
        },
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=44, b=2),
        paper_bgcolor='white',
        plot_bgcolor='white',
    )
    return fig


def make_network(current_entity, visible_entities, tot_E, tot_S, live_mode=False):
    fig = go.Figure()
    vis_set = set(visible_entities)

    for src, dst in SC_EDGES:
        if src not in vis_set or dst not in vis_set:
            continue
        x0, y0 = NODE_POS[src]
        x1, y1 = NODE_POS[dst]
        is_mine = (src == current_entity or dst == current_entity)
        fig.add_trace(go.Scatter(
            x=[x0, x1, None], y=[y0, y1, None],
            mode='lines',
            line=dict(color='#2563eb' if is_mine else '#cbd5e1',
                      width=3.5 if is_mine else 1.5),
            showlegend=False, hoverinfo='none',
        ))

    for ent in ENTITY_ORDER:
        if ent not in vis_set:
            continue
        x, y = NODE_POS[ent]
        tier = SUPPLY_CHAIN[ent]['tier']
        display_name = DISPLAY_NAMES.get(ent, ent)

        if ent == current_entity:
            e_tot = float(tot_E.get(ent, 0.5))
            s_tot = float(tot_S.get(ent, 0.5))
            comp  = 0.5 * e_tot + 0.5 * s_tot
            lbl, col = _risk_level(comp)
            size   = 44
            color  = '#ef4444' if live_mode else col
            text   = (f'<b>{ent}</b><br>E: {e_tot:.3f}<br>'
                      f'S: {s_tot:.3f}<br>Comp: {comp:.3f}')
            if live_mode:
                text = '🔴 LIVE<br>' + text
            hover = (f'<b>{display_name}  ← YOU</b><br>Tier: {tier}<br>'
                     f'E_total={e_tot:.4f}<br>S_total={s_tot:.4f}<br>'
                     f'Composite={comp:.4f}<br>Level:{lbl}'
                     f'{"  🔴 Live" if live_mode else ""}<extra></extra>')
            border_color = '#7f1d1d' if live_mode else '#1e293b'
            border_width = 3
        else:
            size   = 30
            color  = '#94a3b8'
            text   = f'{ent}<br>({tier})<br>🔒'
            hover  = (f'<b>{display_name}</b><br>Tier: {tier}<br>'
                      f'Risk score: 🔒 Confidential<extra></extra>')
            border_color = '#64748b'
            border_width = 1.5

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=size, color=color,
                        line=dict(width=border_width, color=border_color)),
            text=text,
            textposition='bottom center',
            showlegend=False,
            hovertemplate=hover,
            name=ent,
        ))

    for tier_lbl, xp in [('Tier 2', 0.0), ('Tier 1', 1.6),
                          ('Focal', 3.2), ('Downstream', 4.8)]:
        fig.add_annotation(
            x=xp, y=2.35, text=f'<b>{tier_lbl}</b>',
            showarrow=False, font=dict(size=13, color='#1e3a8a'),
            bgcolor='rgba(255,255,255,0.85)', bordercolor='#e2e8f0',
            borderwidth=1, borderpad=3,
        )

    live_tag = '  |  🔴 LIVE OVERRIDE MODE' if live_mode else '  |  🔒 = Risk score confidential'
    y_min = -1.15 if live_mode else -0.95
    fig.update_layout(
        height=440, showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.7, 5.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[y_min, 2.7]),
        margin=dict(l=10, r=10, t=55, b=56),
        title=dict(
            text=f"Supply Chain Network — <b>{current_entity}</b>'s View{live_tag}",
            font=dict(size=14, color='#1e3a8a'),
        ),
        plot_bgcolor='#f8fafc', paper_bgcolor='white',
    )
    return fig


def make_trend_chart(entity, local_E, local_S, days=30, live_mode=False):
    np.random.seed(42 if not live_mode else 99)
    dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]
    noise = 0.09 if live_mode else 0.06
    e_series = [float(np.clip(local_E + np.random.normal(0, noise), 0, 1)) for _ in dates]
    s_series = [float(np.clip(local_S + np.random.normal(0, noise), 0, 1)) for _ in dates]
    for i in range(-7, 0):
        alpha_blend = abs(i) / 7
        e_series[i] = alpha_blend * e_series[i] + (1 - alpha_blend) * local_E
        s_series[i] = alpha_blend * s_series[i] + (1 - alpha_blend) * local_S
    comp_series = [0.5 * e + 0.5 * s for e, s in zip(e_series, s_series)]

    fig = go.Figure()
    colour_map = {
        'Environmental Risk': '#00CC96',
        'Social Risk':        '#AB63FA',
        'Composite Risk':     '#FFA15A',
    }
    for col, series in [('Environmental Risk', e_series),
                        ('Social Risk', s_series),
                        ('Composite Risk', comp_series)]:
        fig.add_trace(go.Scatter(x=dates, y=series, name=col,
                                  mode='lines', line=dict(color=colour_map[col], width=2)))
    fig.add_hline(y=0.70, line_dash='dash', line_color='red',
                  annotation_text='High Risk (0.70)', annotation_position='top right')
    fig.add_hline(y=0.40, line_dash='dash', line_color='orange',
                  annotation_text='Medium Risk (0.40)', annotation_position='top right')
    mode_tag = '  🔴 Live Override' if live_mode else '  (FL-seeded, Step 2 fixed baseline)'
    fig.update_layout(
        height=360,
        title=dict(text=f'Risk Trend — {entity}  (30-day simulated history){mode_tag}',
                   font=dict(size=14)),
        xaxis=dict(title=dict(text='Date', standoff=8)),
        yaxis=dict(title='Risk Score', range=[0, 1]),
        legend=dict(orientation='v', yanchor='middle', y=0.5,
                    xanchor='left', x=1.02, font=dict(size=12),
                    bgcolor='rgba(255,255,255,0.8)', bordercolor='#e5e7eb', borderwidth=1),
        margin=dict(l=45, r=155, t=55, b=45),
        plot_bgcolor='white', paper_bgcolor='white',
    )
    return fig


def make_multiscenario_bar(entity, all_scores, scenario_key, pw, live_vals=None):
    """
    Bar chart comparing M0, M1, M2, M3 (and optionally Live Now) for one entity.
    Uses custom weights pw for consistent comparison.
    all_scores: dict of {scenario_key: scoring_result}
    """
    # Build per-scenario values using custom weights
    scenario_colors = {
        'M0': '#3b82f6',
        'M1': '#ef4444',
        'M2': '#10b981',
        'M3': '#f59e0b',
        'M3-Retrained': '#8b5cf6',
    }
    scenario_labels = {
        'M0': 'M0 Baseline',
        'M1': 'M1 GHG ×2',
        'M2': 'M2 Diversity+',
        'M3': 'M3 Walmart↓',
        'M3-Retrained': 'M3 Retrained',
    }
    dims = ['Environmental', 'Social', 'Composite']
    traces = []
    for sk, sres in all_scores.items():
        if sres is None:
            continue
        E_loc = sres.get('Environmental', {}).get('local_risks', {})
        S_loc = sres.get('Social',        {}).get('local_risks', {})
        tot_E = propagate_risk_custom(E_loc, pw)
        tot_S = propagate_risk_custom(S_loc, pw)
        eV = float(tot_E.get(entity, 0.5))
        sV = float(tot_S.get(entity, 0.5))
        cV = 0.5 * eV + 0.5 * sV
        vals = [eV, sV, cV]
        is_current = (sk == scenario_key)
        traces.append(go.Bar(
            name=scenario_labels.get(sk, sk),
            x=dims, y=vals,
            marker_color=scenario_colors.get(sk, '#94a3b8'),
            marker_line_width=3 if is_current else 0.5,
            marker_line_color='#1e293b' if is_current else 'white',
            opacity=1.0 if is_current else 0.7,
            text=[f'{v:.3f}' for v in vals],
            textposition='outside',
            textfont=dict(size=11),
        ))

    if live_vals is not None:
        traces.append(go.Bar(
            name='🔴 Live Now',
            x=dims,
            y=[live_vals.get('E', 0.5), live_vals.get('S', 0.5), live_vals.get('C', 0.5)],
            marker_color='#f97316',
            text=[f'{v:.3f}' for v in [live_vals.get('E', 0.5), live_vals.get('S', 0.5), live_vals.get('C', 0.5)]],
            textposition='outside',
            textfont=dict(size=11),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode='group',
        height=340,
        title=dict(
            text=f'Two-Step Scenario Comparison — {entity}  (current: {scenario_key})',
            font=dict(size=14),
        ),
        yaxis=dict(title='Risk Score (Total)', range=[0, 1.15]),
        legend=dict(orientation='h', y=-0.28),
        margin=dict(l=45, r=20, t=55, b=70),
        plot_bgcolor='white', paper_bgcolor='white',
    )
    fig.add_hline(y=0.70, line_dash='dash', line_color='red', opacity=0.5)
    fig.add_hline(y=0.40, line_dash='dash', line_color='orange', opacity=0.4)
    return fig


def make_weight_radar(entity, pw):
    w = pw[entity]
    labels = [f'β ({entity[:6]})']
    values = [w['beta']]
    for s, a in w['alpha'].items():
        labels.append(f'α ({s[:8]})')
        values.append(a)
    labels.append(labels[0])
    values.append(values[0])
    fig = go.Figure(go.Scatterpolar(
        r=values, theta=labels, fill='toself',
        fillcolor='rgba(37, 99, 235, 0.15)',
        line=dict(color='#2563eb', width=2),
        marker=dict(size=6, color='#2563eb'),
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9)),
            angularaxis=dict(tickfont=dict(size=10)),
        ),
        height=280,
        margin=dict(l=50, r=50, t=40, b=30),
        title=dict(text='Weight Distribution', font=dict(size=13)),
        paper_bgcolor='white', showlegend=False,
    )
    return fig


# =========================================================================
# Recommendation engine
# =========================================================================

_PRIORITY_STYLE = {
    'HIGH':   {'icon': '🔴', 'bg': '#fef2f2', 'border': '#dc2626', 'label_bg': '#dc2626'},
    'MEDIUM': {'icon': '🟡', 'bg': '#fffbeb', 'border': '#d97706', 'label_bg': '#d97706'},
    'LOW':    {'icon': '🟢', 'bg': '#f0fdf4', 'border': '#16a34a', 'label_bg': '#16a34a'},
    'INFO':   {'icon': '🔵', 'bg': '#eff6ff', 'border': '#2563eb', 'label_bg': '#2563eb'},
}

_REG_MAP = {
    'ghg':           'CSRD (ESRS E1), SEC Climate Disclosure Rule, TCFD',
    'emission':      'CSRD (ESRS E1), SEC Climate Disclosure Rule, TCFD',
    'energy':        'CSRD (ESRS E1/E2), ISO 50001',
    'water':         'CSRD (ESRS E3), CEO Water Mandate',
    'waste':         'CSRD (ESRS E5), Basel Convention',
    'biodiversity':  'CSRD (ESRS E4), Kunming-Montreal Framework',
    'labor':         'CSDDD Art.6, ILO Core Conventions, LkSG §2',
    'health':        'CSDDD Art.6, OSHA Standards, ISO 45001',
    'safety':        'CSDDD Art.6, OSHA Standards, ISO 45001',
    'human rights':  'CSDDD Art.6, UN Guiding Principles (UNGPs), LkSG §2',
    'supply chain':  'CSDDD Art.7, LkSG §3, UK Modern Slavery Act',
    'community':     'CSRD (ESRS S3), IFC Performance Standards',
    'diversity':     'CSRD (ESRS S1), EU Pay Transparency Directive',
    'data privacy':  'GDPR Art.5, CCPA',
    'anti-corrupt':  'CSRD (ESRS G1), FCPA, UK Bribery Act',
}


def _get_reg_ref(category_name):
    cat_lower = category_name.lower()
    for kw, ref in _REG_MAP.items():
        if kw in cat_lower:
            return ref
    return 'CSRD General Disclosure, GRI Standards'


def generate_recommendations(entity, tier, role,
                              E_local, S_local, E_total, S_total, composite,
                              cat_data, pw, scenario_key, upstream_ents):
    import pandas as pd
    recs = []

    # 1. Overall composite risk
    if composite >= 0.70:
        recs.append({'priority': 'HIGH', 'category': 'Overall ESG',
                     'title': f'Composite ESG risk is HIGH ({composite:.3f}) — immediate action required',
                     'detail': (f'Your composite risk score of {composite:.3f} exceeds the 0.70 high-risk threshold. '
                                'Prioritise a cross-functional ESG risk review. '
                                'Consider engaging a third-party auditor for independent verification.'),
                     'regulation': 'CSRD Art.19a, CSDDD Art.6, GRI 3 (Material Topics)'})
    elif composite >= 0.40:
        recs.append({'priority': 'MEDIUM', 'category': 'Overall ESG',
                     'title': f'Composite ESG risk is MEDIUM ({composite:.3f}) — monitoring advised',
                     'detail': (f'Composite risk of {composite:.3f} is in the medium range. '
                                'Establish quarterly ESG KPI reviews and set reduction targets.'),
                     'regulation': 'CSRD Art.19a, TCFD Recommendations, GRI 305'})
    else:
        recs.append({'priority': 'LOW', 'category': 'Overall ESG',
                     'title': f'Composite ESG risk is LOW ({composite:.3f}) — maintain current practices',
                     'detail': ('Current risk profile is below the medium threshold. '
                                'Continue monitoring and share best practices with supply chain partners.'),
                     'regulation': 'CSRD Voluntary Best Practice, GRI 2-29'})

    # 2. E vs S imbalance
    gap = abs(E_total - S_total)
    if gap >= 0.15:
        dominant = 'Environmental' if E_total > S_total else 'Social'
        weaker   = 'Social' if dominant == 'Environmental' else 'Environmental'
        recs.append({'priority': 'MEDIUM', 'category': 'Dimension Balance',
                     'title': f'{dominant} risk ({max(E_total,S_total):.3f}) significantly exceeds {weaker} ({min(E_total,S_total):.3f})',
                     'detail': (f'The {gap:.3f} gap between E and S scores suggests skewed resource allocation. '
                                f'Increase investment in {dominant.lower()} risk controls.'),
                     'regulation': 'CSRD (ESRS E1–E5 + S1–S4), SASB Standards'})

    # 3. Upstream propagation
    e_gap = E_total - E_local
    s_gap = S_total - S_local
    if e_gap >= 0.05 or s_gap >= 0.05:
        recs.append({'priority': 'MEDIUM', 'category': 'Upstream Propagation',
                     'title': 'Upstream suppliers are elevating your total risk score',
                     'detail': (f'E risk increased from local {E_local:.3f} to total {E_total:.3f} (+{e_gap:.3f}); '
                                f'S from {S_local:.3f} to {S_total:.3f} (+{s_gap:.3f}). '
                                f'Upstream: {", ".join(upstream_ents) if upstream_ents else "N/A"}. '
                                'Initiate supplier ESG improvement programmes.'),
                     'regulation': 'CSDDD Art.7, LkSG §3, UK Modern Slavery Act'})

    # 4. Two-Step scenario alerts
    if scenario_key == 'M1':
        recs.append({'priority': 'HIGH', 'category': 'Two-Step Alert — M1',
                     'title': 'Step 2 detected: Nutrien GHG doubled — upstream E risk propagating',
                     'detail': ('Nutrien (Tier 2) GHG emissions doubled in M1. Step 2 fixed-baseline '
                                'scoring captures this cleanly: Nutrien R_local rises, flows to '
                                'Goodyear → Mattel. This illustrates Scope 3 supply chain exposure. '
                                'Establish GHG emission caps in supplier contracts.'),
                     'regulation': 'SEC Climate Rule (Scope 3), CSRD ESRS E1-6, SBTi Supply Chain Target'})
    elif scenario_key == 'M2':
        recs.append({'priority': 'INFO', 'category': 'Two-Step Signal — M2',
                     'title': 'Step 2 dimension isolation confirmed: Deere diversity improved (S only)',
                     'detail': ('Deere workforce diversity improved in M2. Two-Step fixed-baseline correctly '
                                'isolates the change: Social risk decreases; Environmental is unchanged. '
                                'Demonstrates E/S dimension independence under the fixed-baseline scoring.'),
                     'regulation': 'CSRD (ESRS S1), EU Pay Transparency Directive'})
    elif scenario_key == 'M3':
        recs.append({'priority': 'INFO', 'category': 'Two-Step Property — M3',
                     'title': 'Step 2 one-directional flow verified: Walmart GHG improved (downstream only)',
                     'detail': ('Walmart GHG improved in M3, but Mattel\'s total risk is unaffected. '
                                'Risk propagation is strictly upstream → downstream. Downstream changes '
                                'are observable (monitoring) but do not propagate upstream.'),
                     'regulation': 'CSRD ESRS 2 (Value Chain), GRI 308'})
    elif scenario_key == 'M3-Retrained':
        recs.append({'priority': 'INFO', 'category': 'Two-Step Step 1 Update — M3 Retrain',
                     'title': 'New FL round on M3: model "cognition" updated — risk scores reflect new classification',
                     'detail': ('Step 1 re-trained on M3 data. The new global model produces different risk '
                                'scores for the same M3 data because its understanding of "what patterns '
                                'predict risk" has changed. This demonstrates the value of periodic FL '
                                'updates (Step 1) alongside continuous monitoring (Step 2).'),
                     'regulation': 'CSRD Art.19a, CSDDD Art.6'})

    # 5. Category-level flags
    for dim_name in ['Environmental', 'Social']:
        df_dim = cat_data.get(dim_name, pd.DataFrame())
        if df_dim.empty:
            continue
        df_ent = df_dim[df_dim['company'] == entity].copy()
        if df_ent.empty:
            continue
        high_cats = df_ent[df_ent['mean_risk_z'] >= 0.5].sort_values('mean_risk_z', ascending=False)
        for _, row in high_cats.head(3).iterrows():
            cat_name = row['category']
            z_val    = row['mean_risk_z']
            hi_pct   = row['high_risk_pct'] * 100
            priority = 'HIGH' if z_val >= 1.0 else 'MEDIUM'
            recs.append({'priority': priority,
                         'category': f'{dim_name[:3]} — {cat_name}',
                         'title': f'{cat_name} sub-category risk elevated (z={z_val:.3f}, {hi_pct:.0f}% above median)',
                         'detail': (f'{hi_pct:.0f}% of your {cat_name.lower()} metrics are above peer median. '
                                    f'Mean risk z-score: {z_val:.3f}. Conduct targeted gap analysis.'),
                         'regulation': _get_reg_ref(cat_name)})

    # 6. Tier-specific
    if tier == 'Focal':
        recs.append({'priority': 'INFO', 'category': 'Focal Firm Governance',
                     'title': 'As focal company, you bear full supply chain due diligence responsibility',
                     'detail': ('CSDDD and LkSG place primary due diligence obligations on focal firms. '
                                'Ensure Tier 1 suppliers (Goodyear, Sherwin-Williams) submit annual ESG disclosures. '
                                'Consider extending FL-based monitoring to Tier 2 for Scope 3 compliance.'),
                     'regulation': 'CSDDD Art.6–7, LkSG §3–4, CSRD ESRS 2 (Value Chain)'})
    elif tier == 'Tier 2':
        recs.append({'priority': 'INFO', 'category': 'Tier 2 Obligations',
                     'title': 'Proactively share ESG data with Tier 1 to reduce propagated risk',
                     'detail': ('Your ESG performance propagates to Goodyear and ultimately to Mattel. '
                                'Proactive FL participation reduces buyer scrutiny. '
                                'Consider ISO 14001 or CDP verification.'),
                     'regulation': 'CSDDD Art.7, LkSG §3, CDP Supply Chain Programme'})
    elif tier == 'Tier 1':
        recs.append({'priority': 'INFO', 'category': 'Tier 1 Obligations',
                     'title': 'Bridge Tier 2 ESG data to focal firm — your role is critical',
                     'detail': ('You receive upstream risk from Nutrien/Deere and transmit it to Mattel. '
                                'Establish supplier codes of conduct and share FL-hashed ESG metrics.'),
                     'regulation': 'CSDDD Art.6–7, LkSG §3, CSRD ESRS 2 SBM-3'})
    elif tier == 'Downstream':
        recs.append({'priority': 'INFO', 'category': 'Downstream Monitoring',
                     'title': 'Monitor Mattel ESG performance — you can observe but not affect upstream risk',
                     'detail': ('In the Two-Step system, downstream improvements do not propagate upstream '
                                '(by design). Your ESG monitoring of Mattel affects your own Scope 3 '
                                'reporting and reputational risk.'),
                     'regulation': 'CSRD ESRS 2 (Value Chain), SEC Climate Rule (Scope 3), GRI 308'})

    # 7. FL privacy
    recs.append({'priority': 'INFO', 'category': 'FL Privacy Compliance',
                 'title': 'Two-Step FL architecture supports GDPR data minimisation',
                 'detail': ('Step 1 (FL training): only SHA-256 hashed model parameters are transmitted. '
                            'Step 2 (risk scoring): only aggregated risk scores are propagated. '
                            'Raw ESG data never leaves your organisational boundary. '
                            'Satisfies GDPR Art.5(1)(c) and CSDDD proportionality requirements.'),
                 'regulation': 'GDPR Art.5, CSDDD Recital 58, CCPA §1798.100'})

    return recs


def render_recommendation_card(rec):
    p  = rec['priority']
    st_style = _PRIORITY_STYLE.get(p, _PRIORITY_STYLE['INFO'])
    return (
        f"<div style='background:{st_style['bg']};border-left:4px solid {st_style['border']};"
        f"border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.6rem;'>"
        f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem;'>"
        f"<span style='background:{st_style['label_bg']};color:white;border-radius:4px;"
        f"padding:1px 7px;font-size:0.75rem;font-weight:700;'>{p}</span>"
        f"<span style='font-size:0.75rem;color:#64748b;font-weight:600;'>{rec['category']}</span>"
        f"</div>"
        f"<div style='font-weight:700;color:#1e293b;margin-bottom:0.25rem;font-size:0.92rem;'>"
        f"{st_style['icon']} {rec['title']}</div>"
        f"<div style='color:#475569;font-size:0.85rem;margin-bottom:0.3rem;'>{rec['detail']}</div>"
        f"<div style='font-size:0.78rem;color:#64748b;'>"
        f"📋 <b>Regulatory reference:</b> {rec['regulation']}</div>"
        f"</div>"
    )


# =========================================================================
# Category breakdown — adapted for Two-Step (normalize_and_label returns tuple)
# =========================================================================

def compute_category_breakdown(filepath):
    """
    Load ESG data and compute per-company, per-category mean risk_z.
    Uses rolling baseline (Mode B) for display purposes.
    Returns: {dim_name: DataFrame[company, category, mean_risk_z, n_metrics, high_risk_pct]}
    """
    raw_data = load_esg_data(filepath)
    result = {}
    for dim_name, df_raw in raw_data.items():
        # Two-Step: normalize_and_label returns (df, stats) — use rolling baseline (no baseline_stats arg)
        df_norm, _ = normalize_and_label(df_raw, baseline_stats=None)
        if df_norm.empty:
            result[dim_name] = pd.DataFrame()
            continue
        agg = (
            df_norm.groupby(['company', 'category'])
            .agg(
                mean_risk_z  =('risk_z', 'mean'),
                n_metrics    =('risk_z', 'count'),
                high_risk_pct=('label',  'mean'),
            )
            .reset_index()
        )
        result[dim_name] = agg
    return result


def build_live_category_view(cat_data, entity, base_E_local, base_S_local,
                              cur_E_local, cur_S_local):
    if not cat_data:
        return {}
    live_data = {k: v.copy(deep=True) for k, v in cat_data.items()}
    deltas = {
        'Environmental': float(cur_E_local - base_E_local),
        'Social':        float(cur_S_local - base_S_local),
    }
    for dim_name, delta in deltas.items():
        df_dim = live_data.get(dim_name, pd.DataFrame())
        if df_dim.empty:
            continue
        mask = (df_dim['company'] == entity)
        if not mask.any():
            continue
        z_shift   = delta * 2.2
        pct_shift = delta * 0.75
        df_dim.loc[mask, 'mean_risk_z'] = (
            df_dim.loc[mask, 'mean_risk_z'].astype(float) + z_shift
        ).clip(-2.2, 2.2)
        df_dim.loc[mask, 'high_risk_pct'] = (
            df_dim.loc[mask, 'high_risk_pct'].astype(float) + pct_shift
        ).clip(0.0, 1.0)
        live_data[dim_name] = df_dim
    return live_data


def make_category_bar_h(entity, cat_data, dim_name, selected_cats=None):
    df = cat_data.get(dim_name, pd.DataFrame())
    if df.empty:
        return go.Figure()
    df_ent = df[df['company'] == entity].copy()
    if df_ent.empty:
        return go.Figure()
    df_ent = df_ent.sort_values('mean_risk_z', ascending=True)
    z_vals  = df_ent['mean_risk_z'].clip(-2, 2)
    normed  = (z_vals + 2) / 4
    colours = []
    for v in normed:
        if v >= 0.75:   colours.append('#dc2626')
        elif v >= 0.50: colours.append('#d97706')
        elif v >= 0.25: colours.append('#16a34a')
        else:           colours.append('#0ea5e9')
    opacities = [1.0 if (selected_cats is None or not selected_cats or cat in selected_cats)
                 else 0.30 for cat in df_ent['category']]
    hover_texts = [
        f"<b>{row['category']}</b><br>Mean risk_z={row['mean_risk_z']:.4f}<br>"
        f"High-risk metrics={row['high_risk_pct']*100:.1f}%<br># Metrics={int(row['n_metrics'])}"
        for _, row in df_ent.iterrows()
    ]
    fig = go.Figure(go.Bar(
        x=df_ent['mean_risk_z'], y=df_ent['category'], orientation='h',
        marker=dict(color=colours, opacity=opacities, line=dict(width=0.5, color='white')),
        hovertemplate='%{customdata}<extra></extra>', customdata=hover_texts,
        text=[f"{v:.3f}" for v in df_ent['mean_risk_z']], textposition='outside',
        textfont=dict(size=11),
    ))
    dim_colour = '#059669' if dim_name == 'Environmental' else '#7c3aed'
    fig.add_vline(x=0,   line_dash='solid', line_color='#94a3b8', line_width=1)
    fig.add_vline(x=0.5, line_dash='dot',   line_color='#d97706', line_width=1,
                  annotation_text='Medium', annotation_position='top',
                  annotation_font=dict(size=9, color='#d97706'))
    fig.add_vline(x=1.0, line_dash='dot',   line_color='#dc2626', line_width=1,
                  annotation_text='High',   annotation_position='top',
                  annotation_font=dict(size=9, color='#dc2626'))
    fig.update_layout(
        height=max(260, 52 * len(df_ent)),
        title=dict(text=f'{dim_name} Sub-category Risk — {entity}',
                   font=dict(size=13, color=dim_colour)),
        xaxis=dict(title='Mean Risk Z-score', zeroline=False, range=[-2.2, 2.6]),
        yaxis=dict(title='', tickfont=dict(size=11)),
        margin=dict(l=10, r=60, t=48, b=36),
        plot_bgcolor='white', paper_bgcolor='white', showlegend=False,
    )
    return fig


def make_category_heatmap(entity, cat_data, selected_cats=None):
    rows = []
    for dim_name in ['Environmental', 'Social']:
        df = cat_data.get(dim_name, pd.DataFrame())
        if df.empty:
            continue
        df_ent = df[df['company'] == entity]
        for _, r in df_ent.iterrows():
            rows.append({'Dimension': dim_name[:3], 'Category': r['category'],
                         'mean_z': r['mean_risk_z'], 'hi_pct': r['high_risk_pct'] * 100,
                         'n': int(r['n_metrics'])})
    if not rows:
        return go.Figure()
    df_all = pd.DataFrame(rows)
    if selected_cats:
        df_all = df_all[df_all['Category'].isin(selected_cats)]
    if df_all.empty:
        return go.Figure()
    df_all['Label'] = df_all['Dimension'] + ' | ' + df_all['Category']
    df_all = df_all.sort_values(['Dimension', 'mean_z'], ascending=[True, False])
    z_matrix    = [[row['mean_z'], row['hi_pct']] for _, row in df_all.iterrows()]
    text_matrix = [[f"{row['mean_z']:.3f}", f"{row['hi_pct']:.0f}%"] for _, row in df_all.iterrows()]
    fig = go.Figure(go.Heatmap(
        z=z_matrix, x=['Mean Risk Z-score', 'High-Risk Metrics (%)'],
        y=df_all['Label'].tolist(),
        text=text_matrix, texttemplate='%{text}', textfont=dict(size=11),
        colorscale=[[0.00, '#0ea5e9'], [0.30, '#16a34a'],
                    [0.55, '#d97706'], [1.00, '#dc2626']],
        colorbar=dict(title='Risk Level', thickness=12, len=0.8), zmid=0,
    ))
    fig.update_layout(
        height=max(300, 38 * len(df_all)),
        title=dict(text=f'ESG Category Heatmap — {entity}', font=dict(size=13, color='#1e3a8a')),
        xaxis=dict(side='top', tickfont=dict(size=11)),
        yaxis=dict(tickfont=dict(size=10), autorange='reversed'),
        margin=dict(l=10, r=10, t=70, b=20),
        plot_bgcolor='white', paper_bgcolor='white',
    )
    return fig


# =========================================================================
# Cached Two-Step pipeline — runs once per Streamlit session
# =========================================================================

if _HAVE_DASHBOARD:
    @st.cache_resource(show_spinner=False)
    def run_twostep_cached():
        """
        Step 1: FL Training on M0 (once, low-frequency).
        Step 2: Risk scoring on M0/M1/M2/M3 with fixed M0 baseline.
        Also runs Step 1 again on M3 to demonstrate model update.
        All results cached for the session.
        """
        try:
            _dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            _dir = os.getcwd()

        path_m0 = os.path.join(_dir, 'ESG_Raw_Data_Supply_Chain_M0.xlsx')
        path_m1 = os.path.join(_dir, 'ESG_Raw_Data_Supply_Chain_M1.xlsx')
        path_m2 = os.path.join(_dir, 'ESG_Raw_Data_Supply_Chain_M2.xlsx')
        path_m3 = os.path.join(_dir, 'ESG_Raw_Data_Supply_Chain_M3.xlsx')

        # Step 1: Train global model on M0
        fl_result = run_fl_training(path_m0, 'M0', verbose=False)

        # Step 2: Score all available months with FIXED M0 baseline
        scores_m0 = run_risk_scoring(path_m0, 'M0', fl_result, verbose=False)
        scores_m1 = run_risk_scoring(path_m1, 'M1', fl_result, verbose=False) if os.path.exists(path_m1) else None
        scores_m2 = run_risk_scoring(path_m2, 'M2', fl_result, verbose=False) if os.path.exists(path_m2) else None
        scores_m3 = run_risk_scoring(path_m3, 'M3', fl_result, verbose=False) if os.path.exists(path_m3) else None

        # Step 1 again: Re-train on M3 to demonstrate model update
        scores_m3_retrained = None
        if scores_m3 is not None:
            fl_result_m3 = run_fl_training(path_m3, 'M3-Retrain', verbose=False)
            scores_m3_retrained = run_risk_scoring(path_m3, 'M3-Retrained', fl_result_m3, verbose=False)

        return {
            'M0':          scores_m0,
            'M1':          scores_m1,
            'M2':          scores_m2,
            'M3':          scores_m3,
            'M3-Retrained': scores_m3_retrained,
        }

    @st.cache_resource(show_spinner=False)
    def load_category_breakdown_cached():
        """Pre-compute category breakdowns for all available months. Cached per session."""
        try:
            _dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            _dir = os.getcwd()
        paths = {
            'M0': os.path.join(_dir, 'ESG_Raw_Data_Supply_Chain_M0.xlsx'),
            'M1': os.path.join(_dir, 'ESG_Raw_Data_Supply_Chain_M1.xlsx'),
            'M2': os.path.join(_dir, 'ESG_Raw_Data_Supply_Chain_M2.xlsx'),
            'M3': os.path.join(_dir, 'ESG_Raw_Data_Supply_Chain_M3.xlsx'),
        }
        result = {}
        for key, fp in paths.items():
            if os.path.exists(fp):
                result[key] = compute_category_breakdown(fp)
        # M3-Retrained uses same raw data as M3 (only model differs)
        if 'M3' in result:
            result['M3-Retrained'] = result['M3']
        return result

else:
    def run_twostep_cached():
        raise RuntimeError('Streamlit / Plotly not installed.')

    def load_category_breakdown_cached():
        raise RuntimeError('Streamlit / Plotly not installed.')


# =========================================================================
# DASHBOARD MAIN
# =========================================================================

def dashboard_main():
    """Streamlit application for Two-Step FL ESG Risk Dashboard."""
    if not _HAVE_DASHBOARD:
        print('Dashboard requires: pip install streamlit plotly')
        return

    st.set_page_config(
        page_title='FL ESG Risk Dashboard — Two-Step',
        page_icon='🌿',
        layout='wide',
        initial_sidebar_state='expanded',
    )
    st.markdown(DASH_CSS, unsafe_allow_html=True)

    # =========================================================
    # SIDEBAR
    # =========================================================
    st.sidebar.markdown('## 🏢 My Identity')
    current_entity = st.sidebar.selectbox(
        'Select your company',
        options=ENTITY_ORDER,
        index=4,  # default: Mattel (Focal)
        help='Your view is restricted to direct supply chain partners.',
    )

    page = st.sidebar.radio(
        '📑 View',
        ['📊 Risk Dashboard', 'ℹ️ Two-Step System Info'],
        index=0,
        key='page_nav',
    )

    vis_info      = SUPPLY_CHAIN_VIS[current_entity]
    visible_ents  = vis_info['can_see']
    upstream_ents = vis_info['direct_suppliers']
    tier          = SUPPLY_CHAIN[current_entity]['tier']
    role          = SUPPLY_CHAIN[current_entity]['role']
    display_name  = DISPLAY_NAMES.get(current_entity, current_entity)

    # ---- Real-time toggle ----
    st.sidebar.markdown('---')
    st.sidebar.markdown('## 🔄 Data Mode')
    live_mode = st.sidebar.toggle(
        '🔴 Live Override Mode',
        value=False,
        help=(
            'OFF: Risk sliders initialised from Step 2 FL-computed values.\n'
            'ON: Sliders start at 0.500 for fresh real-time self-assessment.'
        ),
        key='live_mode_toggle',
    )
    live_auto         = False
    live_interval_sec = 2.0
    live_drift_sigma  = 0.012

    if live_mode:
        live_auto = st.sidebar.toggle('🔁 Auto Live Drift', value=True, key='live_auto_toggle')
        live_interval_sec = st.sidebar.slider('Refresh interval (s)', 1.0, 10.0, 2.0, 0.5, key='live_interval_sec')
        live_drift_sigma  = st.sidebar.slider('Drift amplitude (σ)', 0.001, 0.050, 0.012, 0.001, key='live_drift_sigma')

    if live_mode:
        st.sidebar.markdown(
            "<div style='background:#fef3c7;border:1.5px solid #f59e0b;border-radius:8px;"
            "padding:0.5rem 0.8rem;color:#78350f;font-size:0.85rem;'>"
            "🔴 <b>Live Override Active</b><br>Sliders start at 0.500. "
            "Drag to set real-time self-reported risk.</div>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown(
            "<div style='background:#f0fdf4;border:1.5px solid #86efac;border-radius:8px;"
            "padding:0.5rem 0.8rem;color:#166534;font-size:0.85rem;'>"
            "🟢 <b>Step 2 FL Data Mode</b><br>Sliders initialised from Step 2 "
            "risk scoring (fixed M0 baseline).</div>", unsafe_allow_html=True)

    # ---- Scenario selector ----
    st.sidebar.markdown('---')
    st.sidebar.markdown('## 📊 Scenario (Step 2 Month)')
    available_scenarios = SCENARIO_OPTIONS  # all defined; unavailable ones handled gracefully
    scenario = st.sidebar.radio('Dataset / Month', available_scenarios, index=0)
    scenario_key = SCENARIO_KEY_MAP[scenario]

    st.sidebar.markdown(
        f"<div class='scenario-box'>"
        f"<b>{scenario_key}</b>: {SCENARIO_DESC.get(scenario_key, '')}</div>",
        unsafe_allow_html=True,
    )

    # ---- Load FL results ----
    with st.spinner('🤖 Running Two-Step FL pipeline (Step 1 on M0, Step 2 on M0–M3)…'):
        try:
            all_scores = run_twostep_cached()
        except Exception as exc:
            st.error(f'Two-Step FL pipeline error: {exc}')
            st.stop()

    results = all_scores.get(scenario_key)
    if results is None:
        st.warning(f'Data file for {scenario_key} not found. Falling back to M0.')
        results = all_scores.get('M0', {})
        scenario_key = 'M0'

    fl_E_loc = results.get('Environmental', {}).get('local_risks', {})
    fl_S_loc = results.get('Social',        {}).get('local_risks', {})
    fl_E_tot = results.get('Environmental', {}).get('total_risks', {})
    fl_S_tot = results.get('Social',        {}).get('total_risks', {})

    # ---- Live-mode state ----
    entering_live = live_mode and (not st.session_state.get('prev_live_mode', False))
    live_e_key = f'live_e_state_{current_entity}_{scenario_key}'
    live_s_key = f'live_s_state_{current_entity}_{scenario_key}'
    live_t_key = f'live_last_update_all_{scenario_key}'

    if live_mode:
        now_ts = time.time()
        for ent in ENTITY_ORDER:
            e_key = f'live_e_state_{ent}_{scenario_key}'
            s_key = f'live_s_state_{ent}_{scenario_key}'
            base_e = float(fl_E_loc.get(ent, 0.5))
            base_s = float(fl_S_loc.get(ent, 0.5))
            if entering_live:
                st.session_state[e_key] = 0.500 if ent == current_entity else base_e
                st.session_state[s_key] = 0.500 if ent == current_entity else base_s
            else:
                if e_key not in st.session_state: st.session_state[e_key] = base_e
                if s_key not in st.session_state: st.session_state[s_key] = base_s

        if entering_live:
            st.session_state[live_t_key] = now_ts
        else:
            last_ts = float(st.session_state.get(live_t_key, now_ts))
            elapsed = max(0.0, now_ts - last_ts)
            if live_auto and elapsed >= live_interval_sec:
                steps = min(5, int(elapsed // live_interval_sec))
                for _ in range(max(1, steps)):
                    for ent in ENTITY_ORDER:
                        e_key = f'live_e_state_{ent}_{scenario_key}'
                        s_key = f'live_s_state_{ent}_{scenario_key}'
                        st.session_state[e_key] = float(np.clip(
                            st.session_state[e_key] + np.random.normal(0, live_drift_sigma), 0.0, 1.0))
                        st.session_state[s_key] = float(np.clip(
                            st.session_state[s_key] + np.random.normal(0, live_drift_sigma), 0.0, 1.0))
                st.session_state[live_t_key] = now_ts

    st.session_state['prev_live_mode'] = live_mode

    # ---- My risk sliders ----
    st.sidebar.markdown('---')
    st.sidebar.markdown('## 📐 My Risk Scores')
    st.sidebar.markdown(
        "<div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;"
        "padding:0.45rem 0.75rem;font-size:0.82rem;color:#1e40af;margin-bottom:0.4rem;'>"
        "<b>💡 Step 2 FL Baseline vs Self-Assessment</b><br>"
        "• <b>Step 2 Baseline</b>: Scored by global model + fixed M0 baseline<br>"
        "• <b>Self-Assessment</b>: Manual real-time override</div>",
        unsafe_allow_html=True,
    )

    fl_E_hint  = round(float(fl_E_loc.get(current_entity, 0.5)), 4)
    fl_S_hint  = round(float(fl_S_loc.get(current_entity, 0.5)), 4)
    default_E  = fl_E_hint if not live_mode else 0.500
    default_S  = fl_S_hint if not live_mode else 0.500

    if live_mode:
        my_E_local = st.sidebar.slider(
            '🔴 Environmental Risk (Local)', 0.000, 1.000, step=0.001, key=live_e_key,
            help=f'Step 2 value: {fl_E_hint:.4f}')
        my_S_local = st.sidebar.slider(
            '🔴 Social Risk (Local)', 0.000, 1.000, step=0.001, key=live_s_key,
            help=f'Step 2 value: {fl_S_hint:.4f}')
        st.sidebar.caption(f'Step 2 reference: E={fl_E_hint:.4f}, S={fl_S_hint:.4f}  |  '
                           f'Auto={"ON" if live_auto else "OFF"}  |  '
                           f'Interval={live_interval_sec:.1f}s  |  σ={live_drift_sigma:.3f}')
    else:
        my_E_local = st.sidebar.slider(
            'Environmental Risk (Local)', 0.000, 1.000, value=default_E, step=0.001,
            help=f'Step 2 baseline: {fl_E_hint:.4f}',
            key=f'slider_my_E_{current_entity}_{scenario_key}')
        my_S_local = st.sidebar.slider(
            'Social Risk (Local)', 0.000, 1.000, value=default_S, step=0.001,
            help=f'Step 2 baseline: {fl_S_hint:.4f}',
            key=f'slider_my_S_{current_entity}_{scenario_key}')
        _e_ov = abs(my_E_local - fl_E_hint) > 0.005
        _s_ov = abs(my_S_local - fl_S_hint) > 0.005
        _tag  = '⚠️ Overridden — ' if (_e_ov or _s_ov) else ''
        st.sidebar.caption(f'{_tag}Step 2 baseline: E={fl_E_hint:.4f}, S={fl_S_hint:.4f}')

    # ---- Propagation weights ----
    st.sidebar.markdown('---')
    st.sidebar.markdown('## ⚖️ Supply Chain Dependency Weights')
    st.sidebar.caption('Enter procurement share (%) for each party. Weights auto-derived.')

    pw        = deepcopy(PROPAGATION_WEIGHTS)
    default_w = PROPAGATION_WEIGHTS[current_entity]

    if not upstream_ents:
        st.sidebar.info(f'{current_entity} has no upstream suppliers.  \nOwn operations = **100%** → β = 1.00')
        my_beta  = 1.0
        my_alpha = {}
    elif len(upstream_ents) == 1:
        u0 = upstream_ents[0]
        default_u0_pct  = round(float(default_w['alpha'].get(u0, 0.5)) * 100)
        u0_pct = st.sidebar.slider(f'📦 {u0} procurement share (%)', 0, 100, value=default_u0_pct, step=1,
                                    key=f'pct_{u0}_{current_entity}')
        own_pct = 100 - u0_pct
        st.sidebar.caption(f'Own operations: **{own_pct}%** (auto)')
        my_beta  = round(own_pct / 100, 4)
        my_alpha = {u0: round(u0_pct / 100, 4)}
    elif len(upstream_ents) == 2:
        u0, u1 = upstream_ents[0], upstream_ents[1]
        default_u0_pct = round(float(default_w['alpha'].get(u0, 0.12)) * 100)
        default_u1_pct = round(float(default_w['alpha'].get(u1, 0.18)) * 100)
        u0_pct = st.sidebar.slider(f'📦 {u0} procurement share (%)', 0, 100, value=default_u0_pct, step=1,
                                    key=f'pct_{u0}_{current_entity}')
        u1_pct = st.sidebar.slider(f'📦 {u1} procurement share (%)', 0, 100, value=default_u1_pct, step=1,
                                    key=f'pct_{u1}_{current_entity}')
        own_pct = 100 - u0_pct - u1_pct
        if own_pct < 0:
            st.sidebar.markdown(
                f"<div style='background:#fef2f2;border:1px solid #fca5a5;border-radius:6px;"
                f"padding:0.35rem 0.6rem;font-size:0.82rem;color:#991b1b;'>"
                f"⚠️ Total exceeds 100% by {-own_pct}% — reduce supplier shares.</div>",
                unsafe_allow_html=True)
            own_pct = 0
        else:
            st.sidebar.caption(f'Own operations: **{own_pct}%** (auto = 100 − {u0_pct} − {u1_pct})')
        my_beta  = round(own_pct / 100, 4)
        my_alpha = {u0: round(u0_pct / 100, 4), u1: round(u1_pct / 100, 4)}
    else:
        n       = len(upstream_ents)
        per_pct = 100 // (n + 1)
        own_pct = 100 - per_pct * n
        my_beta  = round(own_pct / 100, 4)
        my_alpha = {u: round(per_pct / 100, 4) for u in upstream_ents}

    pw[current_entity] = {'beta': my_beta, 'alpha': my_alpha}

    w_sum = my_beta + sum(my_alpha.values())
    if upstream_ents and abs(w_sum - 1.0) < 0.01:
        _conf_lines = [f"&nbsp;&nbsp;{_u}: <b>{round(_a*100)}%</b> → α={_a:.2f}"
                       for _u, _a in my_alpha.items()]
        _conf_lines.append(f"&nbsp;&nbsp;Own: <b>{round(my_beta*100)}%</b> → β={my_beta:.2f}")
        st.sidebar.markdown(
            "<div style='background:#f0fdf4;border:1px solid #86efac;border-radius:6px;"
            "padding:0.45rem 0.75rem;font-size:0.81rem;color:#166534;margin-top:0.4rem;'>"
            "✅ <b>Weights confirmed</b><br>" + "<br>".join(_conf_lines) + "</div>",
            unsafe_allow_html=True)
    elif upstream_ents:
        st.sidebar.markdown(
            f"<div style='background:#fef2f2;border:1px solid #fca5a5;border-radius:6px;"
            f"padding:0.35rem 0.6rem;font-size:0.82rem;color:#991b1b;'>"
            f"⚠️ Shares sum to {round(w_sum*100)}% — must equal 100%.</div>",
            unsafe_allow_html=True)

    # =========================================================
    # COMPUTE PROPAGATED RISKS WITH CUSTOM WEIGHTS
    # =========================================================
    if live_mode:
        adj_E_loc = {ent: float(st.session_state.get(f'live_e_state_{ent}_{scenario_key}', fl_E_loc.get(ent, 0.5)))
                     for ent in ENTITY_ORDER}
        adj_S_loc = {ent: float(st.session_state.get(f'live_s_state_{ent}_{scenario_key}', fl_S_loc.get(ent, 0.5)))
                     for ent in ENTITY_ORDER}
        adj_E_loc[current_entity] = float(my_E_local)
        adj_S_loc[current_entity] = float(my_S_local)
    else:
        adj_E_loc = dict(fl_E_loc)
        adj_S_loc = dict(fl_S_loc)
        adj_E_loc[current_entity] = my_E_local
        adj_S_loc[current_entity] = my_S_local

    try:
        tot_E = propagate_risk_custom(adj_E_loc, pw)
        tot_S = propagate_risk_custom(adj_S_loc, pw)
    except Exception:
        tot_E = fl_E_tot
        tot_S = fl_S_tot

    my_E_total   = float(tot_E.get(current_entity, my_E_local))
    my_S_total   = float(tot_S.get(current_entity, my_S_local))
    my_composite = 0.5 * my_E_total + 0.5 * my_S_total

    lbl_E, col_E = _risk_level(my_E_total)
    lbl_S, col_S = _risk_level(my_S_total)
    lbl_C, col_C = _risk_level(my_composite)

    updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # =========================================================
    # PAGE: TWO-STEP SYSTEM INFO
    # =========================================================
    if page == 'ℹ️ Two-Step System Info':
        st.markdown("<div class='dash-header'>ℹ️ Two-Step FL System Information</div>",
                    unsafe_allow_html=True)
        st.markdown(
            f"<div class='dash-sub'>Two-Step Architecture · Supply Chain Structure · "
            f"Active Weights · Privacy Guarantees — <b>{display_name}</b> ({tier})</div>",
            unsafe_allow_html=True)
        st.caption(f"Viewing as: {display_name} | {tier} | {role} | {updated}")
        st.markdown('---')

        # Two-Step architecture explanation
        st.markdown('### 🏗️ Two-Step Architecture')
        arch_col1, arch_col2 = st.columns(2)
        with arch_col1:
            st.markdown(
                "<div style='background:#eff6ff;border-left:4px solid #3b82f6;"
                "border-radius:8px;padding:0.8rem 1rem;'>"
                "<div class='step1-badge'>Step 1 — Low Frequency</div>"
                "<br><br><b>FL Training (FedAvg)</b><br>"
                "• 7 firms each train local Logistic Regression on their ESG data<br>"
                "• Upload only: coef + intercept (SHA-256 hashed, no raw data)<br>"
                "• Server performs FedAvg → global model<br>"
                "• Output: Global model + baseline stats (rolling, Plan B)<br>"
                "• Triggered: when industry-wide ESG risk landscape changes</div>",
                unsafe_allow_html=True)
        with arch_col2:
            st.markdown(
                "<div style='background:#f0fdf4;border-left:4px solid #10b981;"
                "border-radius:8px;padding:0.8rem 1rem;'>"
                "<div class='step2-badge'>Step 2 — High Frequency</div>"
                "<br><br><b>Risk Scoring & Propagation</b><br>"
                "• Use FIXED baseline from Step 1 (Plan A) for z-score normalization<br>"
                "• Run global model as inference only (no re-training)<br>"
                "• Compute R_local per firm<br>"
                "• Propagate: R_total = β×R_local + Σα×R_upstream_total<br>"
                "• Triggered: monthly (or higher frequency)</div>",
                unsafe_allow_html=True)

        st.markdown('')
        st.markdown('### 📋 Demo Scenarios (M0 → M3 + Retrain)')
        scen_rows = []
        for sk, desc in SCENARIO_DESC.items():
            scen_rows.append({'Scenario': sk, 'Description': desc,
                              'Step': 'Step 1 + Step 2' if sk in ('M0', 'M3-Retrained') else 'Step 2 only'})
        import pandas as _pd_sys
        st.dataframe(_pd_sys.DataFrame(scen_rows), hide_index=True, use_container_width=True)

        st.markdown('---')
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('### 🏗️ Supply Chain Structure')
            for ent in ENTITY_ORDER:
                cfg = SUPPLY_CHAIN[ent]
                up  = ', '.join(cfg['upstream']) if cfg['upstream'] else 'None'
                you = '  ← **YOU**' if ent == current_entity else ''
                vis = '  *(visible to you)*' if ent in visible_ents else ''
                st.markdown(
                    f"- **{DISPLAY_NAMES.get(ent, ent)}** ({cfg['tier']}) &nbsp; upstream: {up}{you}{vis}")
            st.markdown('')
            st.markdown('### 👁️ Visibility Rules')
            st.markdown(
                f"As **{display_name}** ({tier}), you can see:  \n"
                + ',  \n'.join([f"  • {DISPLAY_NAMES.get(e, e)}" for e in visible_ents]))
            st.markdown('')
            st.markdown('### 🔗 Propagation Chain (upstream → downstream)')
            st.markdown(
                "```\n"
                "Nutrien(T2)  ──┐\n"
                "               ├──► Goodyear(T1) ──┐\n"
                "Deere(T2)    ──┘                   ├──► Mattel(Focal) ──┬──► Walmart(Down)\n"
                "                  Sherwin-W(T1)  ──┘                   └──► Target(Down)\n"
                "```")

        with col_b:
            st.markdown('### ⚖️ Your Active Weights')
            w = pw[current_entity]
            st.markdown(f"- β (own weight) = **{w['beta']:.4f}**")
            for s, a in w['alpha'].items():
                st.markdown(f"- α({s}) = **{a:.4f}**")
            total_w = w['beta'] + sum(w['alpha'].values())
            ok = '✅' if abs(total_w - 1.0) < 0.005 else '⚠️'
            st.markdown(f"- **Sum = {total_w:.4f}** {ok}")

            st.markdown('')
            st.markdown('### 🔒 Privacy Guarantees')
            st.markdown(
                "<div style='background:#f1f5f9;border:1.5px solid #94a3b8;"
                "border-radius:8px;padding:0.9rem 1.1rem;'>"
                "<ul style='margin:0;padding-left:1.2rem;'>"
                "<li>Raw ESG data <b>never leaves</b> your organisational boundary</li>"
                "<li>Step 1: only <b>SHA-256 hashed</b> model parameters are transmitted</li>"
                "<li>Step 2: only aggregated risk scores propagate through the chain</li>"
                f"<li>FedAvg aggregation over <b>{FL_ROUNDS} communication rounds</b></li>"
                "<li>Your risk score is <b>invisible</b> to all other entities</li>"
                "</ul></div>",
                unsafe_allow_html=True)

            st.markdown('')
            st.markdown('### 🤖 FL Algorithm Details')
            _sys_rows = [
                {'Parameter': 'Algorithm',             'Value': 'FedAvg (Federated Averaging)'},
                {'Parameter': 'Communication Rounds',  'Value': str(FL_ROUNDS)},
                {'Parameter': 'Local Model',           'Value': 'Logistic Regression (z-score normalised)'},
                {'Parameter': 'Step 1 Normalisation',  'Value': 'Rolling baseline (Plan B) — recomputed each training'},
                {'Parameter': 'Step 2 Normalisation',  'Value': 'Fixed baseline (Plan A) — frozen from Step 1 M0 training'},
                {'Parameter': 'Privacy Mechanism',     'Value': 'SHA-256 parameter hashing (simulation)'},
                {'Parameter': 'ESG Dimensions',        'Value': 'Environmental, Social (Governance excluded)'},
                {'Parameter': 'Risk Propagation',      'Value': 'One-directional: upstream → downstream only'},
                {'Parameter': 'Participating Firms',   'Value': str(len(SUPPLY_CHAIN))},
            ]
            st.dataframe(_pd_sys.DataFrame(_sys_rows), hide_index=True, use_container_width=True)

        st.markdown('---')
        st.caption(f"Two-Step FL ESG Dashboard  |  {display_name} ({tier})  |  "
                   f"FedAvg {FL_ROUNDS} rounds  |  SHA-256 privacy  |  {updated}")
        return

    # =========================================================
    # PAGE: RISK DASHBOARD
    # =========================================================
    st.markdown(
        f"<div class='dash-header'>🌿 Two-Step FL ESG Dashboard — {display_name}</div>",
        unsafe_allow_html=True)
    mode_badge = (
        "<span style='background:#ef4444;color:white;border-radius:6px;"
        "padding:2px 8px;font-size:0.85rem;'>🔴 LIVE OVERRIDE</span>"
        if live_mode else
        "<span style='background:#16a34a;color:white;border-radius:6px;"
        "padding:2px 8px;font-size:0.85rem;'>🟢 STEP 2 FL DATA</span>"
    )
    step_badge = (
        "<span class='step1-badge' style='font-size:0.78rem;'>Step 1: M0 Training</span>&nbsp;"
        "<span class='step2-badge' style='font-size:0.78rem;'>Step 2: "
        + scenario_key + " Scoring</span>"
    )
    st.markdown(
        f"<div class='dash-sub'>{tier} &nbsp;|&nbsp; {role} &nbsp;|&nbsp; "
        f"{step_badge} &nbsp;|&nbsp; {mode_badge}</div>",
        unsafe_allow_html=True)
    st.caption(
        f"Visible network: {len(visible_ents)} entities  |  Updated: {updated}  |  "
        f"FedAvg {FL_ROUNDS} rounds  |  Fixed M0 baseline (Plan A)  |  SHA-256 privacy"
    )

    if live_mode:
        st.markdown(
            "<div class='live-banner'>🔴 <b>Live Override Mode Active</b> — "
            "Risk scores are manually set (not from Step 2 FL pipeline). "
            "Use the sidebar sliders to enter real-time self-reported E and S scores. "
            "Toggle off to return to Step 2-computed values.</div>",
            unsafe_allow_html=True)

    # ── Section 1: My Risk Score ──────────────────────────────────────────
    st.markdown('---')
    st.subheader('📊 My Risk Assessment (Step 2 Output)')
    st.markdown(
        "<div class='privacy-badge'>🔒 <b>Privacy:</b> Only YOUR own risk scores are displayed. "
        "All supply chain partners' scores remain confidential under the federated privacy model. "
        "Step 2 uses the fixed M0 baseline — scores change only when your data changes.</div>",
        unsafe_allow_html=True)
    st.markdown('')

    fl_E_tot_me = float(fl_E_tot.get(current_entity, 0.5))
    fl_S_tot_me = float(fl_S_tot.get(current_entity, 0.5))
    fl_comp_me  = 0.5 * fl_E_tot_me + 0.5 * fl_S_tot_me

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='{'card-live' if live_mode else _card_class(my_composite)}'>"
            f"<b>Composite ESG Risk</b><br>"
            f"<span style='font-size:1.9rem'><b>{my_composite:.4f}</b></span><br>"
            f"<span class='tiny'>{lbl_C}"
            + (f" &nbsp;|&nbsp; FL={fl_comp_me:.4f}" if live_mode else "") +
            f"</span></div>", unsafe_allow_html=True)
    with c2:
        st.markdown(
            f"<div class='{'card-live' if live_mode else _card_class(my_E_total)}'>"
            f"<b>Environmental (Total)</b><br>"
            f"<span style='font-size:1.9rem'><b>{my_E_total:.4f}</b></span><br>"
            f"<span class='tiny'>{lbl_E} &nbsp;|&nbsp; local={my_E_local:.4f}"
            + (f"<br>Step2={fl_E_tot_me:.4f}" if live_mode else "") +
            f"</span></div>", unsafe_allow_html=True)
    with c3:
        st.markdown(
            f"<div class='{'card-live' if live_mode else _card_class(my_S_total)}'>"
            f"<b>Social (Total)</b><br>"
            f"<span style='font-size:1.9rem'><b>{my_S_total:.4f}</b></span><br>"
            f"<span class='tiny'>{lbl_S} &nbsp;|&nbsp; local={my_S_local:.4f}"
            + (f"<br>Step2={fl_S_tot_me:.4f}" if live_mode else "") +
            f"</span></div>", unsafe_allow_html=True)
    with c4:
        st.markdown(
            f"<div class='card-blue'>"
            f"<b>Two-Step FL System</b><br>"
            f"<span style='font-size:1.2rem'>{FL_ROUNDS} rounds</span><br>"
            f"<span class='tiny'>Step1: M0 train &nbsp;|&nbsp; "
            f"Step2: {scenario_key}&nbsp;|&nbsp; Plan A baseline</span></div>",
            unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(make_gauge(my_E_total, 'Environmental Risk (Total)', height=265),
                        use_container_width=True, key=f'gauge_E_{current_entity}_{scenario_key}')
    with g2:
        st.plotly_chart(make_gauge(my_S_total, 'Social Risk (Total)', height=265),
                        use_container_width=True, key=f'gauge_S_{current_entity}_{scenario_key}')
    with g3:
        st.plotly_chart(make_gauge(my_composite, 'Composite ESG Risk', height=265),
                        use_container_width=True, key=f'gauge_C_{current_entity}_{scenario_key}')

    # Propagation formula table
    st.markdown('**📐 How is my Environmental Risk (Total) calculated? (Step 2 formula)**')
    st.caption(
        'Step 2 propagation: R_total = β × R_local + Σ α_j × R_total(upstream_j)  '
        '(fixed M0 baseline for z-score normalisation)')
    w_cur = pw[current_entity]
    _formula_rows = []
    _pct_denom    = my_E_total if abs(my_E_total) > 1e-9 else 1.0
    _own_contrib  = w_cur['beta'] * my_E_local
    _formula_rows.append({'Component': 'Your own risk (β × E_local)',
                           'Weight': f"β={w_cur['beta']:.4f}",
                           'Risk Score': f"E_local={my_E_local:.4f}",
                           'Contribution': f"{_own_contrib:.4f}",
                           'Share': f"{_own_contrib / _pct_denom * 100:.1f}%"})
    for _s, _a in w_cur['alpha'].items():
        _r_s     = float(tot_E.get(_s, 0.5))
        _contrib = _a * _r_s
        _formula_rows.append({'Component': f'Upstream: {_s} (α × E_total)',
                               'Weight': f"α={_a:.4f}",
                               'Risk Score': f"E_total({_s})={_r_s:.4f}",
                               'Contribution': f"{_contrib:.4f}",
                               'Share': f"{_contrib / _pct_denom * 100:.1f}%"})
    _formula_rows.append({'Component': '➜  E_total', 'Weight': '—',
                           'Risk Score': '—', 'Contribution': f"{my_E_total:.4f}", 'Share': '100%'})
    import pandas as _pd_local
    st.dataframe(_pd_local.DataFrame(_formula_rows), hide_index=True, use_container_width=True)

    # ── Section 2: Recommendations ───────────────────────────────────────
    st.markdown('---')
    st.subheader('🎯 Risk Mitigation Recommendations')
    st.caption('Rule-based recommendations from Step 2 risk scores, category breakdown, '
               'tier position, and Two-Step scenario context.')

    try:
        _cat_all_recs = load_category_breakdown_cached()
        _cat_data_recs = _cat_all_recs.get(scenario_key, _cat_all_recs.get('M0', {}))
    except Exception:
        _cat_data_recs = {}

    if live_mode:
        _cat_data_recs = build_live_category_view(
            _cat_data_recs, current_entity, fl_E_hint, fl_S_hint, my_E_local, my_S_local)

    recs = generate_recommendations(
        entity=current_entity, tier=tier, role=role,
        E_local=my_E_local, S_local=my_S_local,
        E_total=my_E_total, S_total=my_S_total, composite=my_composite,
        cat_data=_cat_data_recs, pw=pw, scenario_key=scenario_key,
        upstream_ents=upstream_ents)

    priority_filter = st.multiselect('Filter by priority',
                                      ['HIGH', 'MEDIUM', 'LOW', 'INFO'],
                                      default=['HIGH', 'MEDIUM', 'LOW', 'INFO'],
                                      key=f'rec_filter_{current_entity}')
    recs_filtered = [r for r in recs if r['priority'] in priority_filter]
    from collections import Counter
    counts = Counter(r['priority'] for r in recs)
    rc1, rc2, rc3, rc4 = st.columns(4)
    for col, pri, emoji in zip([rc1, rc2, rc3, rc4], ['HIGH', 'MEDIUM', 'LOW', 'INFO'], ['🔴', '🟡', '🟢', '🔵']):
        with col:
            st.metric(label=f'{emoji} {pri}', value=counts.get(pri, 0))
    st.markdown('')
    if not recs_filtered:
        st.info('No recommendations match the selected priority filter.')
    else:
        for rec in recs_filtered:
            st.markdown(render_recommendation_card(rec), unsafe_allow_html=True)

    # ── Section 3: Weights ────────────────────────────────────────────────
    st.markdown('---')
    st.subheader('⚖️ My Propagation Weights')
    wv_col1, wv_col2 = st.columns([1, 1])
    with wv_col1:
        st.plotly_chart(make_weight_radar(current_entity, pw),
                        use_container_width=True, key=f'weight_radar_{current_entity}')
    with wv_col2:
        st.markdown('**Active Weight Configuration**')
        w = pw[current_entity]
        weight_rows = [{'Component': f'β ({current_entity})', 'Weight': f'{w["beta"]:.4f}',
                        'Meaning': 'Own local risk (Step 2 output)'}]
        for s, a in w['alpha'].items():
            weight_rows.append({'Component': f'α ({s})', 'Weight': f'{a:.4f}',
                                 'Meaning': f'Upstream R_total from {s}'})
        total_w = w['beta'] + sum(w['alpha'].values())
        weight_rows.append({'Component': 'SUM', 'Weight': f'{total_w:.4f}', 'Meaning': '✅ Must equal 1.00'})
        st.dataframe(_pd_local.DataFrame(weight_rows), hide_index=True, use_container_width=True)
        st.markdown(
            f"**Formula:**  R_total = β × R_local + Σ α_j × R_total(upstream_j)  \n"
            f"β={w['beta']:.4f}" +
            ('  |  ' + '  |  '.join([f"α({s})={a:.4f}" for s, a in w['alpha'].items()])
             if w['alpha'] else '  (no upstream suppliers)')
        )

    # ── Section 4: Category Risk Breakdown ───────────────────────────────
    st.markdown('---')
    st.subheader('📋 Category Risk Breakdown')
    st.caption(
        'Per sub-category mean risk z-score for your company (display only — does not change R_local/R_total). '
        'Computed using rolling baseline (Plan B) on the selected month\'s data for reference.'
        + (' Live mode: dynamically adjusted from sidebar sliders.' if live_mode else ''))

    try:
        cat_breakdown_all = load_category_breakdown_cached()
    except Exception as _exc:
        st.warning(f'Category breakdown unavailable: {_exc}')
        cat_breakdown_all = {}

    cat_data      = cat_breakdown_all.get(scenario_key, cat_breakdown_all.get('M0', {}))
    cat_data_view = (build_live_category_view(cat_data, current_entity, fl_E_hint, fl_S_hint,
                                               my_E_local, my_S_local)
                     if live_mode else cat_data)

    all_cats_E, all_cats_S = [], []
    if cat_data_view:
        df_e = cat_data_view.get('Environmental', pd.DataFrame())
        df_s = cat_data_view.get('Social',        pd.DataFrame())
        if not df_e.empty: all_cats_E = sorted(df_e['category'].unique().tolist())
        if not df_s.empty: all_cats_S = sorted(df_s['category'].unique().tolist())

    filt_col1, filt_col2 = st.columns(2)
    with filt_col1:
        sel_cats_E = st.multiselect('🌱 Environmental sub-categories', options=all_cats_E,
                                     default=all_cats_E, key=f'cat_filter_E_{current_entity}')
    with filt_col2:
        sel_cats_S = st.multiselect('👥 Social sub-categories', options=all_cats_S,
                                     default=all_cats_S, key=f'cat_filter_S_{current_entity}')

    view_mode = st.radio('Display style', ['Side-by-side bars (E & S)', 'Combined heatmap'],
                          horizontal=True, key=f'cat_view_{current_entity}')

    if not cat_data_view:
        st.info('Category breakdown data not available.')
    elif view_mode == 'Side-by-side bars (E & S)':
        cb1, cb2 = st.columns(2)
        with cb1:
            fig_e = make_category_bar_h(current_entity, cat_data_view, 'Environmental', sel_cats_E)
            if fig_e.data: st.plotly_chart(fig_e, use_container_width=True,
                                            key=f'cat_bar_E_{current_entity}_{scenario_key}')
            else: st.info('No Environmental category data.')
        with cb2:
            fig_s = make_category_bar_h(current_entity, cat_data_view, 'Social', sel_cats_S)
            if fig_s.data: st.plotly_chart(fig_s, use_container_width=True,
                                            key=f'cat_bar_S_{current_entity}_{scenario_key}')
            else: st.info('No Social category data.')
    else:
        combined_sel = list(set(sel_cats_E) | set(sel_cats_S)) if (sel_cats_E or sel_cats_S) else None
        fig_hm = make_category_heatmap(current_entity, cat_data_view, combined_sel)
        if fig_hm.data:
            st.plotly_chart(fig_hm, use_container_width=True,
                            key=f'cat_hm_{current_entity}_{scenario_key}')
        else: st.info('No category data.')

    with st.expander('📊 Category Risk Summary Table', expanded=False):
        rows_tbl = []
        for dim_name in ['Environmental', 'Social']:
            df_dim = cat_data_view.get(dim_name, pd.DataFrame())
            if df_dim.empty: continue
            df_ent = df_dim[df_dim['company'] == current_entity]
            for _, r in df_ent.iterrows():
                lbl, _ = _risk_level(max(0, min(1, (r['mean_risk_z'] + 2) / 4)))
                rows_tbl.append({'Dimension': dim_name, 'Sub-Category': r['category'],
                                  'Mean Risk Z': round(r['mean_risk_z'], 4),
                                  'High-Risk Metrics': f"{r['high_risk_pct']*100:.1f}%",
                                  '# Metrics': int(r['n_metrics']), 'Risk Level': lbl})
        if rows_tbl:
            st.dataframe(_pd_local.DataFrame(rows_tbl).sort_values(
                ['Dimension', 'Mean Risk Z'], ascending=[True, False]),
                         hide_index=True, use_container_width=True)
        else:
            st.info('No category data available.')

    # ── Section 5: Supply Chain Network ──────────────────────────────────
    st.markdown('---')
    st.subheader('🔗 Supply Chain Network')
    st.caption(
        f"Entities visible to {display_name}: "
        f"**{', '.join([DISPLAY_NAMES.get(e, e) for e in visible_ents])}**  "
        f"|  Grey nodes 🔒 = risk scores confidential")
    st.plotly_chart(
        make_network(current_entity, visible_ents, tot_E, tot_S, live_mode),
        use_container_width=True,
        key=f'network_{current_entity}_{scenario_key}_{live_mode}')

    _all_connected = list(set(upstream_ents + vis_info['direct_customers']))
    if _all_connected:
        st.markdown('---')
        _show_params = st.checkbox(
            '🔍 Show directly connected companies\' FL risk parameters',
            value=False, key=f'show_params_{current_entity}',
            help='You can view structural β and α weights but not neighbours\' risk scores.')
        if _show_params:
            st.subheader('🔍 Connected Companies — Step 2 Risk Parameters')
            with st.expander('📋 FL Risk Parameters table', expanded=True):
                _param_rows = []
                for _ent in ENTITY_ORDER:
                    if _ent not in _all_connected: continue
                    _rel  = 'Supplier ↑' if _ent in upstream_ents else 'Customer ↓'
                    _e_tot = float(fl_E_tot.get(_ent, float('nan')))
                    _s_tot = float(fl_S_tot.get(_ent, float('nan')))
                    _comp  = round(0.5 * _e_tot + 0.5 * _s_tot, 4) if not (_e_tot != _e_tot) else float('nan')
                    def _fmt(v): return f'{v:.4f}' if v == v else '—'
                    _param_rows.append({'Company': _ent, 'Tier': SUPPLY_CHAIN[_ent]['tier'],
                                         'Relationship': _rel,
                                         'E Risk (Step2)': _fmt(_e_tot),
                                         'S Risk (Step2)': _fmt(_s_tot),
                                         'Composite (Step2)': _fmt(_comp)})
                if _param_rows:
                    st.dataframe(_pd_local.DataFrame(_param_rows), hide_index=True, use_container_width=True)
                    st.caption(f'Step 2 risk scores using fixed M0 baseline — scenario: {scenario_key}')

    # ── Section 6: Risk Trend ─────────────────────────────────────────────
    st.markdown('---')
    st.subheader('📈 Risk Trend (30-day Simulated History)')
    st.caption(
        f'Trend seeded from your {"manually-set" if live_mode else "Step 2"} local risk scores '
        f'(E_local={my_E_local:.4f}, S_local={my_S_local:.4f}). '
        + ('Adjust sidebar sliders to see how the trend shifts.' if not live_mode
           else '🔴 Live Override: adjust sliders to simulate different risk scenarios.'))
    st.plotly_chart(
        make_trend_chart(current_entity, my_E_local, my_S_local, live_mode=live_mode),
        use_container_width=True,
        key=f'trend_{current_entity}_{my_E_local:.3f}_{my_S_local:.3f}_{live_mode}')

    # ── Section 7: Multi-Scenario Comparison ─────────────────────────────
    st.markdown('---')
    st.subheader('🔄 Two-Step Scenario Comparison (M0 → M1 → M2 → M3)')
    st.caption(
        'All four Step 2 scenarios compared using your current weight settings. '
        'The highlighted bar is the currently selected scenario. '
        'Key properties to observe: (1) M1: E risk up for Nutrien/Goodyear/Mattel; '
        '(2) M2: S risk down for Deere, E unchanged; '
        '(3) M3: Walmart risk changes, Mattel unchanged; '
        '(4) M3-Retrained: same M3 data but different scores — model updated.'
        + (' 🔴 Live Override: "Live Now" bar added.' if live_mode else ''))

    # Build scenario deltas vs M0
    scores_m0_ref = all_scores.get('M0')
    m0_E_loc = scores_m0_ref.get('Environmental', {}).get('local_risks', {}) if scores_m0_ref else {}
    m0_S_loc = scores_m0_ref.get('Social',        {}).get('local_risks', {}) if scores_m0_ref else {}
    m0_E_cust = propagate_risk_custom(m0_E_loc, pw)
    m0_S_cust = propagate_risk_custom(m0_S_loc, pw)
    m0_E_me   = float(m0_E_cust.get(current_entity, 0.5))
    m0_S_me   = float(m0_S_cust.get(current_entity, 0.5))
    m0_C_me   = 0.5 * m0_E_me + 0.5 * m0_S_me

    cur_E_loc = results.get('Environmental', {}).get('local_risks', {})
    cur_S_loc = results.get('Social',        {}).get('local_risks', {})
    cur_E_cust = propagate_risk_custom(cur_E_loc, pw)
    cur_S_cust = propagate_risk_custom(cur_S_loc, pw)
    cur_E_me   = float(cur_E_cust.get(current_entity, 0.5))
    cur_S_me   = float(cur_S_cust.get(current_entity, 0.5))
    cur_C_me   = 0.5 * cur_E_me + 0.5 * cur_S_me

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        dE = cur_E_me - m0_E_me
        st.metric(label='Environmental Risk (Total)',
                  value=f'{(my_E_total if live_mode else cur_E_me):.4f}',
                  delta=f'vs M0: {dE:+.4f}  ({(dE/m0_E_me*100 if m0_E_me else 0):+.2f}%)',
                  delta_color='inverse')
        st.caption(f'M0: {m0_E_me:.4f}  |  {scenario_key}: {cur_E_me:.4f}')
    with mc2:
        dS = cur_S_me - m0_S_me
        st.metric(label='Social Risk (Total)',
                  value=f'{(my_S_total if live_mode else cur_S_me):.4f}',
                  delta=f'vs M0: {dS:+.4f}  ({(dS/m0_S_me*100 if m0_S_me else 0):+.2f}%)',
                  delta_color='inverse')
        st.caption(f'M0: {m0_S_me:.4f}  |  {scenario_key}: {cur_S_me:.4f}')
    with mc3:
        dC = cur_C_me - m0_C_me
        st.metric(label='Composite Risk',
                  value=f'{(my_composite if live_mode else cur_C_me):.4f}',
                  delta=f'vs M0: {dC:+.4f}  ({(dC/m0_C_me*100 if m0_C_me else 0):+.2f}%)',
                  delta_color='inverse')
        st.caption(f'M0: {m0_C_me:.4f}  |  {scenario_key}: {cur_C_me:.4f}')

    st.plotly_chart(
        make_multiscenario_bar(
            current_entity, all_scores, scenario_key, pw,
            live_vals={'E': my_E_total, 'S': my_S_total, 'C': my_composite} if live_mode else None,
        ),
        use_container_width=True,
        key=f'multi_scen_bar_{current_entity}_{scenario_key}')

    # Propagation chain trace table (Step 2 detail)
    with st.expander('🔍 Step 2 Propagation Chain — All Entities (current scenario)', expanded=False):
        chain_rows = []
        for ent in ENTITY_ORDER:
            e_loc = float(results.get('Environmental', {}).get('local_risks', {}).get(ent, 0.5))
            s_loc = float(results.get('Social',        {}).get('local_risks', {}).get(ent, 0.5))
            e_tot = float(results.get('Environmental', {}).get('total_risks', {}).get(ent, 0.5))
            s_tot = float(results.get('Social',        {}).get('total_risks', {}).get(ent, 0.5))
            comp  = 0.5 * e_tot + 0.5 * s_tot
            lbl, _ = _risk_level(comp)
            ent_tier = SUPPLY_CHAIN[ent]['tier']
            visible_flag = '✅' if ent in visible_ents else '🔒'
            chain_rows.append({
                'Entity':       ent,
                'Tier':         ent_tier,
                'Visible':      visible_flag,
                'E_local':      round(e_loc, 4),
                'E_total':      round(e_tot, 4),
                'S_local':      round(s_loc, 4),
                'S_total':      round(s_tot, 4),
                'Composite':    round(comp, 4),
                'Risk Level':   lbl,
            })
        st.dataframe(_pd_local.DataFrame(chain_rows), hide_index=True, use_container_width=True)
        st.caption(
            'Step 2 uses the global model trained on M0 (Step 1) and the fixed M0 baseline (Plan A). '
            'R_total = β × R_local + Σ α × R_upstream_total  (upstream → downstream only). '
            '🔒 = confidential under federated privacy model (shown here for demo/research purposes).')

    # ── Footer ────────────────────────────────────────────────────────────
    st.markdown('---')
    st.caption(
        f"Two-Step FL ESG Dashboard  |  {display_name} ({tier})  |  "
        f"Step1: FedAvg {FL_ROUNDS} rounds on M0  |  "
        f"Step2: fixed M0 baseline (Plan A)  |  "
        f"Scenario: {scenario_key}  |  "
        f"{'🔴 Live Override' if live_mode else '🟢 Step 2 FL Data'}  |  {updated}"
    )

    # ---- Live auto-refresh ----
    if live_mode and live_auto:
        time.sleep(live_interval_sec)
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()


# =============================================================================
# ENTRY POINT
# =============================================================================

def _in_streamlit_ctx():
    if not _HAVE_DASHBOARD:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == '__main__':
    if _in_streamlit_ctx():
        dashboard_main()
    else:
        main()
