# -*- coding: utf-8 -*-
"""
Federated Learning for ESG Risk Assessment in Supply Chains
Design Science Research — System Demonstration

Algorithm: FedAvg (Federated Averaging)
Local Model: Logistic Regression on z-score normalized ESG metrics
Privacy: SHA-256 parameter hashing (simulation)
Dimensions: Environmental (E), Social (S) — Governance excluded

Created on Sun Mar 15 2026
@author: lulux
"""

import numpy as np
import pandas as pd
import hashlib
import os
import warnings
from copy import deepcopy
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# SUPPLY CHAIN CONFIGURATION
# =============================================================================

# Directed supply chain graph (upstream → downstream)
#   Tier 2: Nutrien, Deere  (both supply Goodyear only)
#     |
#   Tier 1: Goodyear (has upstream), Sherwin-Williams (independent, no upstream)
#     |
#   Focal:  Mattel
#     |
#   Downstream: Walmart, Target  (upstream = Mattel)

SUPPLY_CHAIN = {
    'Nutrien':          {'tier': 'Tier 2',      'role': 'Fertilizer Supplier',    'upstream': []},
    'Deere':            {'tier': 'Tier 2',      'role': 'Agricultural Equipment', 'upstream': []},
    'Goodyear':         {'tier': 'Tier 1',      'role': 'Rubber Factory',         'upstream': ['Nutrien', 'Deere']},
    'Sherwin-Williams': {'tier': 'Tier 1',      'role': 'Paint Manufacturer',     'upstream': []},
    'Mattel':           {'tier': 'Focal',       'role': 'Toy Manufacturer',       'upstream': ['Goodyear', 'Sherwin-Williams']},
    'Walmart':          {'tier': 'Downstream',  'role': 'Retailer 1',            'upstream': ['Mattel']},
    'Target':           {'tier': 'Downstream',  'role': 'Retailer 2',            'upstream': ['Mattel']},
}

# Risk propagation weights: R_i^total = Sum a_ij * R_j^total + B_i * R_i^local
# a_ij = w_ij (no decay function in this demo)
# Normalization: Sum a_ij + B_i = 1.0 for each entity
#
# WEIGHT SUMMARY TABLE:
# +--------------------+--------+------+-------------------------------------+--------+
# | Entity             | Tier   | B_i  | Upstream a_ij                       | Sum    |
# +--------------------+--------+------+-------------------------------------+--------+
# | Nutrien            | Tier 2 | 1.00 | (none)                              | 1.00   |
# | Deere              | Tier 2 | 1.00 | (none)                              | 1.00   |
# | Goodyear           | Tier 1 | 0.50 | Nutrien=0.20, Deere=0.30            | 1.00   |
# | Sherwin-Williams   | Tier 1 | 1.00 | (none - independent)                | 1.00   |
# | Mattel             | Focal  | 0.70 | Goodyear=0.12, Sherwin-W=0.18       | 1.00   |
# | Walmart            | Down.  | 0.99 | Mattel=0.01                         | 1.00   |
# | Target             | Down.  | 0.98 | Mattel=0.02                         | 1.00   |
# +--------------------+--------+------+-------------------------------------+--------+
#
# Goodyear:  B=0.50, upstream share: Nutrien 40%, Deere 60% of (1-0.50)=0.50
# Mattel:    B=0.70, upstream share: Goodyear 40%, Sherwin-W 60% of (1-0.70)=0.30
# Sherwin-W: B=1.00, no upstream suppliers in this demo
# Walmart:   B=0.99, upstream share: Mattel 100% of (1-0.99)=0.01
# Target:    B=0.98, upstream share: Mattel 100% of (1-0.98)=0.02

PROPAGATION_WEIGHTS = {
    'Nutrien':          {'beta': 1.0, 'alpha': {}},
    'Deere':            {'beta': 1.0, 'alpha': {}},
    'Goodyear': {
        'beta': 0.5,
        'alpha': {'Nutrien': 0.4 * 0.5, 'Deere': 0.6 * 0.5}   # 0.20, 0.30
    },
    'Sherwin-Williams': {'beta': 1.0, 'alpha': {}},              # independent, no upstream
    'Mattel': {
        'beta': 0.7,
        'alpha': {'Goodyear': 0.4 * 0.3, 'Sherwin-Williams': 0.6 * 0.3}  # 0.12, 0.18
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

# Excel column name → internal company name
COMPANY_COLUMNS = {
    'Nutrien\n(NTR)':          'Nutrien',
    'Deere\n(DE)':             'Deere',
    'Goodyear\n(GT)':          'Goodyear',
    'Sherwin-Williams\n(SHW)': 'Sherwin-Williams',
    'Mattel\n(MAT)':           'Mattel',
    'Walmart\n(WMT)':          'Walmart',
    'Target\n(TGT)':           'Target',
}

# Display ordering for output
ENTITY_ORDER = ['Nutrien', 'Deere', 'Goodyear', 'Sherwin-Williams',
                'Mattel', 'Walmart', 'Target']

# Keywords identifying metrics where HIGHER value = LOWER risk (invert for scoring)
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

# Keywords for metrics where higher = higher risk (default, no inversion)
# These are listed for documentation only; metrics not matching GOOD are treated as risk
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
    """Parse raw ESG metric value from Excel into a float.
    Handles: ~prefix, commas, %, $, Yes/No/Partial, CDP grades, ratios (X:1).
    Returns np.nan for unparseable or missing values.
    """
    if pd.isna(val):
        return np.nan

    val_str = str(val).strip()

    if val_str.upper() in ('N/D', 'N/R', 'NAN', 'NONE', '-', ''):
        return np.nan

    # Yes / No / Partial
    val_lower = val_str.lower()
    if val_lower.startswith('yes'):
        return 1.0
    if val_lower.startswith('no'):
        return 0.0
    if val_lower.startswith('partial'):
        return 0.5

    # CDP letter grades
    grade_map = {
        'A': 0.95, 'A-': 0.85, 'B+': 0.75, 'B': 0.65, 'B-': 0.55,
        'C+': 0.45, 'C': 0.35, 'C-': 0.25, 'D+': 0.15, 'D': 0.10,
        'D-': 0.05, 'F': 0.0
    }
    if val_str in grade_map:
        return grade_map[val_str]

    # Clean prefix
    clean = val_str.replace('~', '').replace(',', '').replace('$', '').strip()

    # Remove parenthetical notes like "(reported)"
    if '(' in clean:
        clean = clean[:clean.index('(')].strip()

    # Ratios like "85:1"
    if ':1' in clean:
        try:
            return float(clean.replace(':1', ''))
        except ValueError:
            return np.nan

    # Percentages
    if clean.endswith('%'):
        try:
            return float(clean[:-1]) / 100.0
        except ValueError:
            return np.nan

    # Numeric
    try:
        return float(clean)
    except ValueError:
        return np.nan


def load_esg_data(filepath):
    """Load Environmental and Social data from Excel.
    Returns dict: {dimension_name: DataFrame with columns
    [company, category, indicator, unit, value, is_good]}.
    Halcyon Agri (Tier 2+) is excluded.
    """
    sheets = {
        'Environmental': 'Environmental Raw Data',
        'Social':        'Social Raw Data',
    }
    results = {}

    for dim_name, sheet_name in sheets.items():
        df_raw = pd.read_excel(filepath, sheet_name=sheet_name, header=None)

        # Row 0 = sheet title (merged cell), Row 1 = company headers, Row 2 = tiers
        # Data starts at row 3
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

            # Skip pure-text description rows
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


def normalize_and_label(df_dim):
    """Z-score normalize each indicator across companies.
    Direction-adjust so that higher z = higher risk.
    Label: 1 if risk_z > 0 (above-median risk), else 0.
    """
    records = []

    for indicator, group in df_dim.groupby('indicator'):
        if len(group) < 3:
            continue
        values = group['value'].values
        mu = np.mean(values)
        sigma = np.std(values)
        if sigma < 1e-12:
            continue

        is_good = group['is_good'].iloc[0]

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

    return pd.DataFrame(records)


def build_features(df_norm):
    """Build per-company feature matrices for logistic regression.
    Each training sample = one ESG indicator for one firm.
    Features: [risk_z, sub-category one-hot encoding].
    Returns: {company: (X, y)}, list_of_categories.
    """
    categories = sorted(df_norm['category'].unique())
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    n_cats = len(categories)

    company_data = {}
    for company, grp in df_norm.groupby('company'):
        X_rows, y_rows = [], []
        for _, row in grp.iterrows():
            feat = np.zeros(1 + n_cats)
            feat[0] = row['risk_z']
            feat[1 + cat_to_idx[row['category']]] = 1.0
            X_rows.append(feat)
            y_rows.append(row['label'])
        company_data[company] = (np.array(X_rows), np.array(y_rows))

    return company_data, categories


# =============================================================================
# FEDERATED LEARNING — CLIENT
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
        """Train logistic regression locally.
        If global_params provided, warm-start from the global model.
        """
        self.model = LogisticRegression(
            max_iter=300, C=1.0, solver='lbfgs', random_state=42,
        )

        # Ensure both classes present (required by sklearn)
        y_train = y.copy()
        if len(np.unique(y_train)) < 2:
            y_train[0] = 1 - y_train[0]

        if global_params is not None:
            # Fit once to initialise internal structures, then override with global
            self.model.fit(X, y_train)
            self.model.coef_ = global_params['coef'].copy()
            self.model.intercept_ = global_params['intercept'].copy()
            # Re-fit (warm start from global params)
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

    def predict_risk_score(self, X):
        """Average P(high_risk) across all local metrics → R_i^local."""
        if self.model is None:
            return 0.5
        return float(np.mean(self.model.predict_proba(X)[:, 1]))


# =============================================================================
# FEDERATED LEARNING — SERVER
# =============================================================================

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
    R_i^total = Σ_{j ∈ S_i} α_ij · R_j^total + β_i · R_i^local
    Processed in tier order: Tier 2 → Tier 1 → Focal → Downstream.
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
# FL PIPELINE (reusable for multiple datasets)
# =============================================================================

def run_fl_pipeline(filepath, scenario_label, verbose=True):
    """Run the full FL pipeline on a given dataset.
    Returns dict with keys: dimension_results (containing local_risks, total_risks per dim).
    """
    if verbose:
        print_header(f"SCENARIO: {scenario_label}")
        print(f"  Data file: {os.path.basename(filepath)}")

    # -- Load data --
    raw_data = load_esg_data(filepath)

    if verbose:
        print_header("DATA LOADING & PREPROCESSING")
        for dim, df in raw_data.items():
            print(f"  {dim:15s}: {df['indicator'].nunique():3d} indicators, "
                  f"{df['company'].nunique()} companies, {len(df)} data points")

    # -- Normalise, label, build features --
    dimension_results = {}

    for dim_name, df_raw in raw_data.items():
        df_norm = normalize_and_label(df_raw)
        company_data, categories = build_features(df_norm)

        if verbose:
            print(f"\n  {dim_name} - sub-categories: {categories}")
            for company in ENTITY_ORDER:
                if company in company_data:
                    X, y = company_data[company]
                    print(f"    {company:20s}: {len(y):3d} samples  "
                          f"(high-risk: {int(y.sum()):2d}, low-risk: {int(len(y) - y.sum()):2d})")

        dimension_results[dim_name] = {
            'company_data': company_data,
            'categories':   categories,
        }

    # -- FL training loop (per dimension) --
    for dim_name, dim_info in dimension_results.items():
        if verbose:
            print_header(f"FL TRAINING - {dim_name.upper()} DIMENSION", char='-')
            print(f"  Model: Logistic Regression | Rounds: {FL_ROUNDS}")
        company_data = dim_info['company_data']
        n_features = list(company_data.values())[0][0].shape[1]

        # Create clients
        clients = {}
        for entity in ENTITY_ORDER:
            if entity in company_data:
                cfg = SUPPLY_CHAIN[entity]
                clients[entity] = FLClient(entity, cfg['tier'], cfg['role'], n_features)

        server = FLServer()
        global_params = None
        prev_coef_norm = None

        for rnd in range(1, FL_ROUNDS + 1):
            if verbose:
                print(f"\n  -- Round {rnd:2d}/{FL_ROUNDS} --")
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

        # -- Parameter transmission chain --
        if verbose:
            print(f"\n  {'-' * 66}")
            print(f"  PARAMETER TRANSMISSION CHAIN - {dim_name.upper()}")
            print(f"  {'-' * 66}")

            # Tier 2 -> Goodyear only (Nutrien & Deere supply Goodyear, not Sherwin-Williams)
            for t2 in ('Nutrien', 'Deere'):
                if t2 in clients and 'Goodyear' in clients:
                    h = clients[t2].hash_params()
                    print(f"    {t2:20s} (Tier 2) --> {'Goodyear':20s} (Tier 1)  | {h[:20]}...")

            # Tier 1 -> Focal
            for t1 in ('Goodyear', 'Sherwin-Williams'):
                if t1 in clients and 'Mattel' in clients:
                    h = clients[t1].hash_params()
                    print(f"    {t1:20s} (Tier 1) --> {'Mattel':20s} (Focal)  | {h[:20]}...")

            # Focal -> Downstream (risk propagation)
            if 'Mattel' in clients:
                h = clients['Mattel'].hash_params()
                for ds in ('Walmart', 'Target'):
                    if ds in clients:
                        print(f"    {'Mattel':20s} (Focal)  --> {ds:20s} (Down.)  | {h[:20]}...")

            # Downstream -> Focal (monitoring, reverse direction)
            for ds in ('Walmart', 'Target'):
                if ds in clients and 'Mattel' in clients:
                    h = clients[ds].hash_params()
                    print(f"    {ds:20s} (Down.)  --> {'Mattel':20s} (Focal)  | {h[:20]}... [MONITORING]")

        # -- Local risk scores --
        if verbose:
            print(f"\n  {'-' * 66}")
            print(f"  LOCAL RISK SCORES  R_i^local - {dim_name.upper()}")
            print(f"  {'-' * 66}")

        local_risks = {}
        for entity in ENTITY_ORDER:
            if entity in clients:
                X, y = company_data[entity]
                score = clients[entity].predict_risk_score(X)
                local_risks[entity] = score
                if verbose:
                    tier = SUPPLY_CHAIN[entity]['tier']
                    bar = print_bar(score)
                    print(f"    {entity:20s} ({tier:12s}): {score:.4f}  |{bar}|")

        # -- Propagated risk scores --
        if verbose:
            print(f"\n  {'-' * 66}")
            print(f"  PROPAGATED RISK  R_i^total = Sum a_ij*R_j^total + B_i*R_i^local")
            print(f"  {'-' * 66}")

        total_risks = propagate_risk(local_risks)

        if verbose:
            for entity in ENTITY_ORDER:
                if entity not in total_risks:
                    continue
                r_tot = total_risks[entity]
                r_loc = local_risks.get(entity, 0.5)
                w = PROPAGATION_WEIGHTS[entity]
                tier = SUPPLY_CHAIN[entity]['tier']
                bar = print_bar(r_tot)

                parts = [f"B*R_local = {w['beta']:.2f}*{r_loc:.4f} = {w['beta'] * r_loc:.4f}"]
                for s, a in w['alpha'].items():
                    r_s = total_risks.get(s, 0.5)
                    parts.append(f"a({s})*R_total = {a:.2f}*{r_s:.4f} = {a * r_s:.4f}")

                print(f"    {entity:20s} ({tier:12s}): R_total = {r_tot:.4f}  |{bar}|")
                print(f"      = {' + '.join(parts)}")

        dim_info['local_risks'] = local_risks
        dim_info['total_risks'] = total_risks

    # -- Composite ESG risk --
    if verbose:
        print_header("COMPOSITE ESG RISK SCORES  (50% Environmental + 50% Social)")
        for entity in ENTITY_ORDER:
            tier = SUPPLY_CHAIN[entity]['tier']
            e = dimension_results.get('Environmental', {}).get('total_risks', {}).get(entity, 0.5)
            s = dimension_results.get('Social', {}).get('total_risks', {}).get(entity, 0.5)
            composite = 0.5 * e + 0.5 * s
            bar = print_bar(composite)
            print(f"  {entity:20s} ({tier:12s}):  E={e:.4f}  S={s:.4f}  "
                  f"-> Composite={composite:.4f}  |{bar}|")

        # -- Focal firm due-diligence view --
        print_header("FOCAL FIRM DUE-DILIGENCE VIEW  (Mattel)")
        e_total = dimension_results.get('Environmental', {}).get('total_risks', {})
        s_total = dimension_results.get('Social', {}).get('total_risks', {})
        e_local = dimension_results.get('Environmental', {}).get('local_risks', {})
        s_local = dimension_results.get('Social', {}).get('local_risks', {})

        print("  Entity                  Tier          E_local  E_total  S_local  S_total  Composite")
        print("  " + "-" * 90)
        for entity in ENTITY_ORDER:
            tier = SUPPLY_CHAIN[entity]['tier']
            el = e_local.get(entity, 0.5)
            et = e_total.get(entity, 0.5)
            sl = s_local.get(entity, 0.5)
            st = s_total.get(entity, 0.5)
            comp = 0.5 * et + 0.5 * st
            flag = " <-- FOCAL" if entity == 'Mattel' else ""
            monitor = " [MON]" if tier == 'Downstream' else ""
            print(f"  {entity:20s}  {tier:12s}  {el:7.4f}  {et:7.4f}  "
                  f"{sl:7.4f}  {st:7.4f}  {comp:9.4f}{flag}{monitor}")

    return dimension_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print_header("FEDERATED LEARNING FOR ESG RISK ASSESSMENT IN SUPPLY CHAINS")
    print("  Design Science Research - System Demonstration")
    print("  Algorithm: FedAvg | Privacy: SHA-256 Hashing (Simulation)")

    # -- Supply chain structure --
    print_header("SUPPLY CHAIN STRUCTURE")
    for entity in ENTITY_ORDER:
        cfg = SUPPLY_CHAIN[entity]
        up = ', '.join(cfg['upstream']) if cfg['upstream'] else 'None'
        print(f"  {cfg['tier']:12s} | {entity:20s} | {cfg['role']:25s} | Upstream: {up}")

    # -- Weight verification --
    print_header("RISK PROPAGATION WEIGHTS  (a_ij = w_ij, no decay)")
    for entity in ENTITY_ORDER:
        w = PROPAGATION_WEIGHTS[entity]
        parts = [f"a({s})={a:.2f}" for s, a in w['alpha'].items()]
        alpha_str = ', '.join(parts) if parts else 'N/A'
        total = w['beta'] + sum(w['alpha'].values())
        print(f"  {entity:20s} | B = {w['beta']:.2f} | {alpha_str:40s} | Sum = {total:.2f}")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # =========================================================================
    # SCENARIO A: Baseline (original data)
    # =========================================================================
    filepath_baseline = os.path.join(script_dir, 'ESG_Raw_Data_Supply_Chain.xlsx')
    results_baseline = run_fl_pipeline(
        filepath_baseline,
        scenario_label="BASELINE (Original ESG Data)",
        verbose=True,
    )

    # =========================================================================
    # SCENARIO B: Risk Added (Nutrien GHG emissions doubled)
    # =========================================================================
    filepath_risk = os.path.join(script_dir, 'ESG_Raw_Data_Supply_Chain - Risk Added.xlsx')
    results_risk = run_fl_pipeline(
        filepath_risk,
        scenario_label="RISK ADDED (Nutrien GHG Emissions 2x)",
        verbose=True,
    )

    # =========================================================================
    # COMPARATIVE ANALYSIS — Environmental Risk Impact
    # =========================================================================
    print_header("COMPARATIVE ANALYSIS: ENVIRONMENTAL RISK IMPACT")
    print("  Change: Nutrien (Tier 2) GHG Emissions doubled (E1-E4)")
    print("  Focus:  How upstream risk change propagates to Focal firm (Mattel)")

    dim = 'Environmental'
    base_local = results_baseline.get(dim, {}).get('local_risks', {})
    base_total = results_baseline.get(dim, {}).get('total_risks', {})
    risk_local = results_risk.get(dim, {}).get('local_risks', {})
    risk_total = results_risk.get(dim, {}).get('total_risks', {})

    # -- Local risk comparison --
    print(f"\n  {'-' * 78}")
    print(f"  LOCAL ENVIRONMENTAL RISK (R_i^local) - Baseline vs Risk-Added")
    print(f"  {'-' * 78}")
    print(f"  {'Entity':20s}  {'Tier':12s}  {'Baseline':>10s}  {'RiskAdded':>10s}  {'Delta':>10s}  {'Change%':>10s}")
    print(f"  {'-' * 78}")
    for entity in ENTITY_ORDER:
        tier = SUPPLY_CHAIN[entity]['tier']
        bl = base_local.get(entity, 0.5)
        ra = risk_local.get(entity, 0.5)
        delta = ra - bl
        pct = (delta / bl * 100) if bl != 0 else 0.0
        marker = " ***" if abs(pct) > 1.0 else ""
        print(f"  {entity:20s}  {tier:12s}  {bl:10.4f}  {ra:10.4f}  {delta:+10.4f}  {pct:+9.2f}%{marker}")

    # -- Propagated risk comparison --
    print(f"\n  {'-' * 78}")
    print(f"  PROPAGATED ENVIRONMENTAL RISK (R_i^total) - Baseline vs Risk-Added")
    print(f"  {'-' * 78}")
    print(f"  {'Entity':20s}  {'Tier':12s}  {'Baseline':>10s}  {'RiskAdded':>10s}  {'Delta':>10s}  {'Change%':>10s}")
    print(f"  {'-' * 78}")
    for entity in ENTITY_ORDER:
        tier = SUPPLY_CHAIN[entity]['tier']
        bl = base_total.get(entity, 0.5)
        ra = risk_total.get(entity, 0.5)
        delta = ra - bl
        pct = (delta / bl * 100) if bl != 0 else 0.0
        marker = " ***" if abs(pct) > 1.0 else ""
        print(f"  {entity:20s}  {tier:12s}  {bl:10.4f}  {ra:10.4f}  {delta:+10.4f}  {pct:+9.2f}%{marker}")

    # -- Focal firm (Mattel) detailed breakdown --
    print(f"\n  {'-' * 78}")
    print(f"  FOCAL FIRM (MATTEL) ENVIRONMENTAL RISK DECOMPOSITION")
    print(f"  {'-' * 78}")

    for label, lr, tr in [("Baseline", base_local, base_total),
                           ("Risk-Added", risk_local, risk_total)]:
        w = PROPAGATION_WEIGHTS['Mattel']
        r_loc = lr.get('Mattel', 0.5)
        parts = [f"B*R_local = {w['beta']:.2f}*{r_loc:.4f} = {w['beta'] * r_loc:.4f}"]
        for s, a in w['alpha'].items():
            r_s = tr.get(s, 0.5)
            parts.append(f"a({s})*R_total = {a:.2f}*{r_s:.4f} = {a * r_s:.4f}")
        print(f"\n  [{label}]")
        print(f"    Mattel R_total = {tr.get('Mattel', 0.5):.4f}")
        print(f"      = {' + '.join(parts)}")

    mattel_base = base_total.get('Mattel', 0.5)
    mattel_risk = risk_total.get('Mattel', 0.5)
    mattel_delta = mattel_risk - mattel_base
    mattel_pct = (mattel_delta / mattel_base * 100) if mattel_base != 0 else 0.0

    print(f"\n  MATTEL ENVIRONMENTAL RISK CHANGE:")
    print(f"    Baseline:   {mattel_base:.4f}")
    print(f"    Risk-Added: {mattel_risk:.4f}")
    print(f"    Delta:      {mattel_delta:+.4f}  ({mattel_pct:+.2f}%)")

    # -- Risk propagation path trace --
    print(f"\n  {'-' * 78}")
    print(f"  RISK PROPAGATION PATH: Full Supply Chain")
    print(f"  {'-' * 78}")
    print(f"  Nutrien(T2) --+")
    print(f"                +--> Goodyear(T1) --+--> Mattel(Focal) --+--> Walmart(Down)")
    print(f"  Deere(T2)   --+                   |                    +--> Target(Down)")
    print(f"                   Sherwin-W(T1) ----+")

    for label, tr in [("Baseline", base_total), ("Risk-Added", risk_total)]:
        print(f"\n  [{label}]")
        nut = tr.get('Nutrien', 0.5)
        dee = tr.get('Deere', 0.5)
        gy  = tr.get('Goodyear', 0.5)
        sw  = tr.get('Sherwin-Williams', 0.5)
        mat = tr.get('Mattel', 0.5)
        wmt = tr.get('Walmart', 0.5)
        tgt = tr.get('Target', 0.5)
        print(f"    Nutrien(T2)={nut:.4f} -+-> Goodyear(T1)={gy:.4f} -+-> Mattel={mat:.4f} -+-> Walmart={wmt:.4f}")
        print(f"    Deere(T2)  ={dee:.4f} -+                          |                      +-> Target ={tgt:.4f}")
        print(f"                             Sherwin-W(T1)={sw:.4f} -+")

    # -- Summary --
    print_header("FL SYSTEM SUMMARY")
    print(f"  Algorithm:             FedAvg (Federated Averaging)")
    print(f"  Communication Rounds:  {FL_ROUNDS}")
    print(f"  Participating Firms:   {len(SUPPLY_CHAIN)}")
    print(f"  Privacy Mechanism:     SHA-256 parameter hashing (simulation)")
    print(f"  ESG Dimensions:        Environmental, Social (Governance excluded)")
    print(f"  Local Model:           Logistic Regression (z-score normalised features)")
    print(f"  Risk Propagation:      a_ij = w_ij (no decay function)")
    print(f"  Downstream Handling:   Participate in local FL training; used for downstream monitoring only, not aggregated into focal upstream risk")
    print()
    print(f"  Scenario Comparison:")
    print(f"    A) Baseline:   Original ESG data")
    print(f"    B) Risk-Added: Nutrien GHG emissions doubled (E1-E4)")
    print(f"    Result:        Focal firm (Mattel) environmental risk changed by {mattel_pct:+.2f}%")
    print(f"                   demonstrating upstream risk propagation through FL system")
    print()
    print(f"  +-------------------------------------------------------------------+")
    print(f" |  KEY PRIVACY GUARANTEE                                    |")
    print(f" |  Raw ESG data never leaves the participating firm.        |")
    print(f" |  Model parameters are exchanged for federated aggregation;|")
    print(f" |  SHA-256 hashes provide integrity demonstration.          |")
    print(f"  +-------------------------------------------------------------------+")
    print("=" * 80)


# =============================================================================
# =============================================================================
# DASHBOARD
# =============================================================================
# =============================================================================
#
# Run with:  streamlit run FL03152025_Dashboard.py
# CLI mode:  python  FL03152025_Dashboard.py
#
# Visibility rules (federated privacy model):
#   Nutrien          → can see: Nutrien, Goodyear
#   Deere            → can see: Deere, Goodyear
#   Goodyear         → can see: Nutrien, Deere, Goodyear, Mattel
#   Sherwin-Williams → can see: Sherwin-Williams, Mattel
#   Mattel           → can see: Goodyear, Sherwin-Williams, Mattel, Walmart, Target
#   Walmart          → can see: Mattel, Walmart
#   Target           → can see: Mattel, Target
#
# Privacy rules (federated privacy model):
#   • Each entity sees ONLY its OWN risk scores
#   • All other entities' risk scores are shown as 🔒 Confidential
#   • Own local risk score can be adjusted via sliders (self-assessment)
#   • Propagation weights β and α can be adjusted via sliders
#
# Real-time toggle:
#   • FL Data Mode (default): sliders initialised from FL-computed values
#   • Live Override Mode: sliders start at 0.500 for fresh self-assessment
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
# Each entity can only see itself and its DIRECT supply chain neighbours.
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

# ---- Node positions (x, y) for supply chain network diagram -------------
NODE_POS = {
    'Nutrien':          (0.0, 1.6),
    'Deere':            (0.0, 0.4),
    'Goodyear':         (1.6, 1.6),
    'Sherwin-Williams': (1.6, 0.4),
    'Mattel':           (3.2, 1.0),
    'Walmart':          (4.8, 1.6),
    'Target':           (4.8, 0.4),
}

# ---- Supply chain directed edges ----------------------------------------
SC_EDGES = [
    ('Nutrien',   'Goodyear'),
    ('Deere',     'Goodyear'),
    ('Goodyear',  'Mattel'),
    ('Sherwin-Williams', 'Mattel'),
    ('Mattel',    'Walmart'),
    ('Mattel',    'Target'),
]

# ---- Display names (for Goodyear full name) --------------------------------
DISPLAY_NAMES = {
    'Goodyear': 'Goodyear Tire & Rubber',
}

# ---- CSS ----------------------------------------------------------------
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
</style>
"""


# =========================================================================
# Helper: propagate risk with CUSTOM weight dict
# =========================================================================

def propagate_risk_custom(local_risks, pw):
    """
    Re-run the propagation formula using a custom weight dict pw.
    pw has the same structure as PROPAGATION_WEIGHTS.
    """
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
    """Return (label, hex_colour) for a risk score."""
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
    """Smooth green → yellow → red semi-circular gauge."""
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
    """
    Draw the supply chain network diagram.
    • Current entity: coloured by composite risk, shows E/S numbers.
    • All other visible entities: grey with 🔒 confidential badge.
    • Invisible entities: not drawn at all.
    """
    fig = go.Figure()
    vis_set = set(visible_entities)

    # ---- Edges ----
    for src, dst in SC_EDGES:
        if src not in vis_set or dst not in vis_set:
            continue
        x0, y0 = NODE_POS[src]
        x1, y1 = NODE_POS[dst]
        is_mine = (src == current_entity or dst == current_entity)
        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(
                color='#2563eb' if is_mine else '#cbd5e1',
                width=3.5 if is_mine else 1.5,
            ),
            showlegend=False,
            hoverinfo='none',
        ))

    # ---- Nodes ----
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
            text   = (f'<b>{ent}</b><br>'
                      f'E: {e_tot:.3f}<br>'
                      f'S: {s_tot:.3f}<br>'
                      f'Composite: {comp:.3f}')
            if live_mode:
                text = '🔴 LIVE<br>' + text
            hover = (f'<b>{display_name}  ← YOU</b><br>'
                     f'Tier: {tier}<br>'
                     f'E_total = {e_tot:.4f}<br>'
                     f'S_total = {s_tot:.4f}<br>'
                     f'Composite = {comp:.4f}<br>'
                     f'Level: {lbl}{"  🔴 Live" if live_mode else ""}<extra></extra>')
            border_color = '#7f1d1d' if live_mode else '#1e293b'
            border_width = 3
        else:
            size   = 30
            color  = '#94a3b8'
            text   = f'{ent}<br>({tier})<br>🔒'
            hover  = (f'<b>{display_name}</b><br>'
                      f'Tier: {tier}<br>'
                      f'Risk score: 🔒 Confidential<br>'
                      f'(not visible to {current_entity})<extra></extra>')
            border_color = '#64748b'
            border_width = 1.5

        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(
                size=size,
                color=color,
                line=dict(width=border_width, color=border_color),
            ),
            text=text,
            textposition='bottom center',
            showlegend=False,
            hovertemplate=hover,
            name=ent,
        ))

    # ---- Tier headers ----
    for tier_lbl, xp in [('Tier 2', 0.0), ('Tier 1', 1.6),
                          ('Focal', 3.2), ('Downstream', 4.8)]:
        fig.add_annotation(
            x=xp, y=2.35,
            text=f'<b>{tier_lbl}</b>',
            showarrow=False,
            font=dict(size=13, color='#1e3a8a'),
            bgcolor='rgba(255,255,255,0.85)',
            bordercolor='#e2e8f0',
            borderwidth=1,
            borderpad=3,
        )

    live_tag = '  |  🔴 LIVE OVERRIDE MODE' if live_mode else '  |  🔒 = Risk score confidential'
    # Keep multi-line node labels fully visible inside the chart frame.
    y_min = -1.15 if live_mode else -0.95
    fig.update_layout(
        height=440,
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[-0.7, 5.5]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   range=[y_min, 2.7]),
        margin=dict(l=10, r=10, t=55, b=56),
        title=dict(
            text=(f"Supply Chain Network — <b>{current_entity}</b>'s View{live_tag}"),
            font=dict(size=14, color='#1e3a8a'),
        ),
        plot_bgcolor='#f8fafc',
        paper_bgcolor='white',
    )
    return fig


def make_trend_chart(entity, local_E, local_S, days=30, live_mode=False):
    """
    30-day simulated risk trend seeded from FL local risk scores.
    In live mode the series is noisier (reflecting real-time uncertainty).
    """
    np.random.seed(42 if not live_mode else 99)
    dates = [datetime.now() - timedelta(days=i) for i in range(days, 0, -1)]

    noise = 0.09 if live_mode else 0.06
    e_series = [float(np.clip(local_E + np.random.normal(0, noise), 0, 1))
                for _ in dates]
    s_series = [float(np.clip(local_S + np.random.normal(0, noise), 0, 1))
                for _ in dates]

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
        fig.add_trace(go.Scatter(
            x=dates, y=series, name=col,
            mode='lines',
            line=dict(color=colour_map[col], width=2),
        ))

    fig.add_hline(y=0.70, line_dash='dash', line_color='red',
                  annotation_text='High Risk (0.70)',
                  annotation_position='top right')
    fig.add_hline(y=0.40, line_dash='dash', line_color='orange',
                  annotation_text='Medium Risk (0.40)',
                  annotation_position='top right')

    mode_tag = '  🔴 Live Override' if live_mode else '  (FL-seeded)'
    fig.update_layout(
        height=360,
        title=dict(text=f'Risk Trend — {entity}  (30-day simulated history){mode_tag}',
                   font=dict(size=14)),
        xaxis=dict(title=dict(text='Date', standoff=8)),
        yaxis=dict(title='Risk Score', range=[0, 1]),
        legend=dict(
            orientation='v',
            yanchor='middle', y=0.5,
            xanchor='left',   x=1.02,
            font=dict(size=12),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='#e5e7eb',
            borderwidth=1,
        ),
        margin=dict(l=45, r=155, t=55, b=45),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    return fig


def make_scenario_bar(entity, res_base, res_risk, live_vals=None,
                      precomp_base_E=None, precomp_base_S=None,
                      precomp_risk_E=None, precomp_risk_S=None):
    """Bar chart comparing Baseline vs Risk-Added (and optional Live Now) for one entity.
    If precomp_* values are provided (custom-weight propagation), they override res_base/res_risk.
    """
    dims = ['Environmental', 'Social', 'Composite']

    bE = precomp_base_E if precomp_base_E is not None else res_base.get('Environmental', {}).get('total_risks', {}).get(entity, 0.5)
    bS = precomp_base_S if precomp_base_S is not None else res_base.get('Social',        {}).get('total_risks', {}).get(entity, 0.5)
    rE = precomp_risk_E if precomp_risk_E is not None else res_risk.get('Environmental', {}).get('total_risks', {}).get(entity, 0.5)
    rS = precomp_risk_S if precomp_risk_S is not None else res_risk.get('Social',        {}).get('total_risks', {}).get(entity, 0.5)

    base_vals = [bE, bS, 0.5 * bE + 0.5 * bS]
    risk_vals = [rE, rS, 0.5 * rE + 0.5 * rS]

    traces = [
        go.Bar(
            name='Baseline',
            x=dims, y=base_vals,
            marker_color='#3b82f6',
            text=[f'{v:.3f}' for v in base_vals],
            textposition='outside',
            textfont=dict(size=11, color='#1d4ed8'),
        ),
        go.Bar(
            name='Risk Added',
            x=dims, y=risk_vals,
            marker_color='#ef4444',
            text=[f'{v:.3f}' for v in risk_vals],
            textposition='outside',
            textfont=dict(size=11, color='#b91c1c'),
        ),
    ]
    if live_vals is not None:
        live_E = float(live_vals.get('E', 0.5))
        live_S = float(live_vals.get('S', 0.5))
        live_C = float(live_vals.get('C', 0.5))
        _live_vals_list = [live_E, live_S, live_C]
        traces.append(go.Bar(
            name='Live Now',
            x=dims,
            y=_live_vals_list,
            marker_color='#f59e0b',
            text=[f'{v:.3f}' for v in _live_vals_list],
            textposition='outside',
            textfont=dict(size=11, color='#92400e'),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        barmode='group',
        height=320,
        title=dict(
            text=(
                f'Scenario Comparison — {entity}  '
                f'(Baseline vs Nutrien GHG ×2{" vs Live Now" if live_vals is not None else ""})'
            ),
            font=dict(size=14),
        ),
        yaxis=dict(title='Risk Score (Total)', range=[0, 1.12]),
        legend=dict(orientation='h', y=-0.25),
        margin=dict(l=45, r=20, t=55, b=65),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )
    fig.add_hline(y=0.70, line_dash='dash', line_color='red', opacity=0.5)
    return fig


def make_weight_radar(entity, pw):
    """Radar chart showing weight distribution for the current entity."""
    w = pw[entity]
    labels = [f'β ({entity[:6]})']
    values = [w['beta']]
    for s, a in w['alpha'].items():
        labels.append(f'α ({s[:8]})')
        values.append(a)

    # Close the polygon
    labels.append(labels[0])
    values.append(values[0])

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=labels,
        fill='toself',
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
        paper_bgcolor='white',
        showlegend=False,
    )
    return fig


# =========================================================================
# Recommendation engine (rule-based, no LLM)
# =========================================================================

# Priority colours / icons
_PRIORITY_STYLE = {
    'HIGH':   {'icon': '🔴', 'bg': '#fef2f2', 'border': '#dc2626', 'label_bg': '#dc2626'},
    'MEDIUM': {'icon': '🟡', 'bg': '#fffbeb', 'border': '#d97706', 'label_bg': '#d97706'},
    'LOW':    {'icon': '🟢', 'bg': '#f0fdf4', 'border': '#16a34a', 'label_bg': '#16a34a'},
    'INFO':   {'icon': '🔵', 'bg': '#eff6ff', 'border': '#2563eb', 'label_bg': '#2563eb'},
}

# Regulatory references keyed by ESG category keyword
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


def _get_reg_ref(category_name: str) -> str:
    """Return the most relevant regulatory reference for a category."""
    cat_lower = category_name.lower()
    for kw, ref in _REG_MAP.items():
        if kw in cat_lower:
            return ref
    return 'CSRD General Disclosure, GRI Standards'


def generate_recommendations(
    entity, tier, role,
    E_local, S_local, E_total, S_total, composite,
    cat_data, pw, scenario_key, upstream_ents,
):
    """
    Rule-based recommendation engine.
    Returns list of dicts: {priority, category, title, detail, regulation}
    Priority: 'HIGH' | 'MEDIUM' | 'LOW' | 'INFO'
    """
    import pandas as pd
    recs = []

    # ------------------------------------------------------------------ #
    # 1. Overall composite risk level
    # ------------------------------------------------------------------ #
    if composite >= 0.70:
        recs.append({
            'priority':   'HIGH',
            'category':   'Overall ESG',
            'title':      f'Composite ESG risk is HIGH ({composite:.3f}) — immediate action required',
            'detail':     (
                f'Your composite risk score of {composite:.3f} exceeds the 0.70 high-risk threshold. '
                'Prioritise a cross-functional ESG risk review covering both Environmental and Social dimensions. '
                'Consider engaging a third-party auditor for independent verification.'
            ),
            'regulation': 'CSRD Art.19a, CSDDD Art.6, GRI 3 (Material Topics)',
        })
    elif composite >= 0.40:
        recs.append({
            'priority':   'MEDIUM',
            'category':   'Overall ESG',
            'title':      f'Composite ESG risk is MEDIUM ({composite:.3f}) — monitoring advised',
            'detail':     (
                f'Composite risk of {composite:.3f} is in the medium range. '
                'Establish quarterly ESG KPI reviews and set reduction targets for the highest-risk sub-categories. '
                'Engage upstream suppliers on shared risk reduction programmes.'
            ),
            'regulation': 'CSRD Art.19a, TCFD Recommendations, GRI 305',
        })
    else:
        recs.append({
            'priority':   'LOW',
            'category':   'Overall ESG',
            'title':      f'Composite ESG risk is LOW ({composite:.3f}) — maintain current practices',
            'detail':     (
                'Current risk profile is below the medium threshold. '
                'Continue monitoring and consider sharing best practices with supply chain partners '
                'to lift the overall supply chain ESG performance.'
            ),
            'regulation': 'CSRD Voluntary Best Practice, GRI 2-29 (Stakeholder Engagement)',
        })

    # ------------------------------------------------------------------ #
    # 2. E vs S imbalance
    # ------------------------------------------------------------------ #
    gap = abs(E_total - S_total)
    if gap >= 0.15:
        dominant = 'Environmental' if E_total > S_total else 'Social'
        weaker   = 'Social' if dominant == 'Environmental' else 'Environmental'
        recs.append({
            'priority':   'MEDIUM',
            'category':   'Dimension Balance',
            'title':      f'{dominant} risk ({max(E_total,S_total):.3f}) significantly exceeds {weaker} ({min(E_total,S_total):.3f})',
            'detail':     (
                f'The {gap:.3f} gap between E and S scores suggests resource allocation is skewed. '
                f'Increase investment in {dominant.lower()} risk controls. '
                f'A balanced ESG approach is increasingly expected by CSRD and investor ESG rating agencies (MSCI, Sustainalytics).'
            ),
            'regulation': 'CSRD (ESRS E1–E5 + S1–S4), SASB Standards',
        })

    # ------------------------------------------------------------------ #
    # 3. Local vs total risk gap (upstream propagation effect)
    # ------------------------------------------------------------------ #
    e_gap = E_total - E_local
    s_gap = S_total - S_local
    if e_gap >= 0.05 or s_gap >= 0.05:
        recs.append({
            'priority':   'MEDIUM',
            'category':   'Upstream Propagation',
            'title':      'Upstream suppliers are elevating your total risk score',
            'detail':     (
                f'Your Environmental risk increased from local {E_local:.3f} to total {E_total:.3f} '
                f'(+{e_gap:.3f}) and Social from {S_local:.3f} to {S_total:.3f} (+{s_gap:.3f}) '
                f'due to upstream risk propagation from: {", ".join(upstream_ents) if upstream_ents else "N/A"}. '
                'Initiate supplier ESG improvement programmes and consider contractual ESG clauses.'
            ),
            'regulation': 'CSDDD Art.7 (Indirect Business Relationships), LkSG §3, UK Modern Slavery Act',
        })
    elif e_gap <= -0.05 or s_gap <= -0.05:
        recs.append({
            'priority':   'INFO',
            'category':   'Upstream Propagation',
            'title':      'Upstream suppliers are helping reduce your total risk',
            'detail':     (
                'Your upstream suppliers\' strong ESG performance is bringing down your total risk scores. '
                'Document these supplier relationships as a competitive advantage in CSRD sustainability reports.'
            ),
            'regulation': 'CSRD Art.19a (Value Chain), CSDDD Art.5',
        })

    # ------------------------------------------------------------------ #
    # 4. Scenario impact warning (Risk Added scenario)
    # ------------------------------------------------------------------ #
    if scenario_key == 'Risk Added':
        recs.append({
            'priority':   'HIGH',
            'category':   'Scenario Alert',
            'title':      'Risk Added scenario active: Nutrien GHG emissions doubled',
            'detail':     (
                'This scenario simulates Nutrien (Tier 2) doubling GHG emissions. '
                'The propagated increase in your Environmental risk score illustrates Scope 3 supply chain exposure. '
                'Establish GHG emission caps in supplier contracts and consider diversifying Tier 2 suppliers '
                'to reduce concentration risk.'
            ),
            'regulation': 'SEC Climate Disclosure Rule (Scope 3), CSRD ESRS E1-6, SBTi Supply Chain Target',
        })

    # ------------------------------------------------------------------ #
    # 5. Category-level high-risk flags
    # ------------------------------------------------------------------ #
    for dim_name in ['Environmental', 'Social']:
        df_dim = cat_data.get(dim_name, pd.DataFrame())
        if df_dim.empty:
            continue
        df_ent = df_dim[df_dim['company'] == entity].copy()
        if df_ent.empty:
            continue

        # Find categories where mean_risk_z > 0.5 (above-median risk)
        high_cats = df_ent[df_ent['mean_risk_z'] >= 0.5].sort_values('mean_risk_z', ascending=False)
        for _, row in high_cats.head(3).iterrows():
            cat_name = row['category']
            z_val    = row['mean_risk_z']
            hi_pct   = row['high_risk_pct'] * 100
            priority = 'HIGH' if z_val >= 1.0 else 'MEDIUM'
            reg_ref  = _get_reg_ref(cat_name)
            recs.append({
                'priority':   priority,
                'category':   f'{dim_name[:3]} — {cat_name}',
                'title':      f'{cat_name} sub-category risk is elevated (z={z_val:.3f}, {hi_pct:.0f}% metrics above median)',
                'detail':     (
                    f'{hi_pct:.0f}% of your {cat_name.lower()} metrics are above the supply chain median. '
                    f'Mean risk z-score is {z_val:.3f}. '
                    'Conduct a targeted gap analysis, set time-bound improvement targets, '
                    'and disclose progress in your next sustainability report.'
                ),
                'regulation': reg_ref,
            })

    # ------------------------------------------------------------------ #
    # 6. Tier-specific governance recommendations
    # ------------------------------------------------------------------ #
    if tier == 'Focal':
        recs.append({
            'priority':   'INFO',
            'category':   'Focal Firm Governance',
            'title':      'As focal company, you are responsible for full supply chain due diligence',
            'detail':     (
                'CSDDD and LkSG place primary due diligence obligations on focal/lead firms. '
                'Ensure your Tier 1 suppliers (Goodyear, Sherwin-Williams) submit annual ESG disclosures. '
                'Consider extending FL-based risk monitoring to Tier 2 (Nutrien, Deere) for Scope 3 compliance.'
            ),
            'regulation': 'CSDDD Art.6–7, LkSG §3–4, CSRD ESRS 2 (Value Chain)',
        })
    elif tier == 'Tier 2':
        recs.append({
            'priority':   'INFO',
            'category':   'Tier 2 Supplier Obligations',
            'title':      'Proactively share ESG data with Tier 1 customers to reduce propagated risk',
            'detail':     (
                'Your ESG performance directly propagates to Goodyear and ultimately to Mattel. '
                'Proactive ESG disclosure and FL participation reduces buyer scrutiny and contract risk. '
                'Consider obtaining ISO 14001 or CDP verification to signal credibility.'
            ),
            'regulation': 'CSDDD Art.7, LkSG §3, CDP Supply Chain Programme',
        })
    elif tier == 'Tier 1':
        recs.append({
            'priority':   'INFO',
            'category':   'Tier 1 Supplier Obligations',
            'title':      'Bridge Tier 2 ESG data to focal firm — your role is critical',
            'detail':     (
                'As a Tier 1 supplier, you receive upstream risk from Nutrien/Deere and transmit it to Mattel. '
                'Establish supplier codes of conduct for your own upstream suppliers '
                'and share aggregated (FL-hashed) ESG metrics with the focal company.'
            ),
            'regulation': 'CSDDD Art.6–7, LkSG §3, CSRD ESRS 2 SBM-3',
        })
    elif tier == 'Downstream':
        recs.append({
            'priority':   'INFO',
            'category':   'Downstream Monitoring',
            'title':      'Monitor Mattel ESG performance as a supply chain risk signal',
            'detail':     (
                'As a downstream retailer, your supplier ESG risk exposure (Mattel) affects your '
                'Scope 3 emissions and reputational risk. '
                'Include supplier ESG scores in vendor assessment frameworks and disclose in your own CSRD report.'
            ),
            'regulation': 'CSRD ESRS 2 (Value Chain), SEC Climate Rule (Scope 3), GRI 308',
        })

    # ------------------------------------------------------------------ #
    # 7. Privacy / FL governance
    # ------------------------------------------------------------------ #
    recs.append({
        'priority':   'INFO',
        'category':   'FL Privacy Compliance',
        'title':      'Federated Learning architecture supports GDPR data minimisation principle',
        'detail':     (
            'The FL system transmits only SHA-256 hashed model parameters — raw ESG data never leaves '
            'your organisational boundary. This satisfies GDPR Art.5(1)(c) (data minimisation) and '
            'supports CSDDD\'s requirement for proportionate due diligence without exposing sensitive '
            'commercial information to supply chain partners.'
        ),
        'regulation': 'GDPR Art.5, CSDDD Recital 58 (Proportionality), CCPA §1798.100',
    })

    return recs


def render_recommendation_card(rec):
    """Render a single recommendation as an HTML card."""
    p  = rec['priority']
    st = _PRIORITY_STYLE.get(p, _PRIORITY_STYLE['INFO'])
    return (
        f"<div style='background:{st['bg']};border-left:4px solid {st['border']};"
        f"border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.6rem;'>"
        f"<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.3rem;'>"
        f"<span style='background:{st['label_bg']};color:white;border-radius:4px;"
        f"padding:1px 7px;font-size:0.75rem;font-weight:700;'>{p}</span>"
        f"<span style='font-size:0.75rem;color:#64748b;font-weight:600;'>{rec['category']}</span>"
        f"</div>"
        f"<div style='font-weight:700;color:#1e293b;margin-bottom:0.25rem;font-size:0.92rem;'>"
        f"{st['icon']} {rec['title']}</div>"
        f"<div style='color:#475569;font-size:0.85rem;margin-bottom:0.3rem;'>{rec['detail']}</div>"
        f"<div style='font-size:0.78rem;color:#64748b;'>"
        f"📋 <b>Regulatory reference:</b> {rec['regulation']}</div>"
        f"</div>"
    )


# =========================================================================
# Category breakdown helpers
# =========================================================================

def compute_category_breakdown(filepath):
    """
    Load ESG data and compute per-company, per-category mean risk_z.
    Returns dict: {dim_name: DataFrame[company, category, mean_risk_z, n_metrics, high_risk_pct]}
    Reuses FL core functions load_esg_data + normalize_and_label (no recalculation).
    """
    raw_data = load_esg_data(filepath)
    result = {}
    for dim_name, df_raw in raw_data.items():
        df_norm = normalize_and_label(df_raw)
        if df_norm.empty:
            result[dim_name] = pd.DataFrame()
            continue
        agg = (
            df_norm.groupby(['company', 'category'])
            .agg(
                mean_risk_z  = ('risk_z', 'mean'),
                n_metrics    = ('risk_z', 'count'),
                high_risk_pct= ('label',  'mean'),   # fraction of metrics labelled high-risk
            )
            .reset_index()
        )
        result[dim_name] = agg
    return result


def build_live_category_view(
    cat_data,
    entity,
    base_E_local,
    base_S_local,
    cur_E_local,
    cur_S_local,
):
    """
    Build a live-adjusted category view for one entity.

    We do not have live raw metric streams per sub-category. In live mode we
    therefore map the current local-risk delta (vs FL baseline) into a smooth
    shift on category aggregates for the selected entity only.
    """
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

        z_shift = delta * 2.2
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
    """
    Horizontal bar chart: mean risk_z per sub-category for one entity.
    Bars are coloured by risk level; selected_cats highlighted.
    """
    import pandas as pd
    df = cat_data.get(dim_name, pd.DataFrame())
    if df.empty:
        return go.Figure()

    df_ent = df[df['company'] == entity].copy()
    if df_ent.empty:
        return go.Figure()

    df_ent = df_ent.sort_values('mean_risk_z', ascending=True)

    # Clip z-scores to [−2, 2] for display then normalise to [0, 1]
    z_vals  = df_ent['mean_risk_z'].clip(-2, 2)
    normed  = (z_vals + 2) / 4          # map [−2,2] → [0,1] for colour scale

    colours = []
    for v in normed:
        if v >= 0.75:
            colours.append('#dc2626')   # red  — high risk
        elif v >= 0.50:
            colours.append('#d97706')   # amber — medium
        elif v >= 0.25:
            colours.append('#16a34a')   # green — low
        else:
            colours.append('#0ea5e9')   # blue  — very low

    # Highlight selected categories with a border / opacity
    opacities = []
    for cat in df_ent['category']:
        if selected_cats is None or not selected_cats or cat in selected_cats:
            opacities.append(1.0)
        else:
            opacities.append(0.30)

    hover_texts = [
        f"<b>{row['category']}</b><br>"
        f"Mean risk_z = {row['mean_risk_z']:.4f}<br>"
        f"High-risk metrics = {row['high_risk_pct']*100:.1f}%<br>"
        f"# Metrics = {int(row['n_metrics'])}"
        for _, row in df_ent.iterrows()
    ]

    fig = go.Figure(go.Bar(
        x          = df_ent['mean_risk_z'],
        y          = df_ent['category'],
        orientation= 'h',
        marker     = dict(color=colours, opacity=opacities, line=dict(width=0.5, color='white')),
        hovertemplate = '%{customdata}<extra></extra>',
        customdata = hover_texts,
        text       = [f"{v:.3f}" for v in df_ent['mean_risk_z']],
        textposition = 'outside',
        textfont   = dict(size=11),
    ))

    dim_colour = '#059669' if dim_name == 'Environmental' else '#7c3aed'
    fig.add_vline(x=0,    line_dash='solid', line_color='#94a3b8', line_width=1)
    fig.add_vline(x=0.5,  line_dash='dot',   line_color='#d97706', line_width=1,
                  annotation_text='Medium', annotation_position='top',
                  annotation_font=dict(size=9, color='#d97706'))
    fig.add_vline(x=1.0,  line_dash='dot',   line_color='#dc2626', line_width=1,
                  annotation_text='High',   annotation_position='top',
                  annotation_font=dict(size=9, color='#dc2626'))

    fig.update_layout(
        height      = max(260, 52 * len(df_ent)),
        title       = dict(
            text    = f'{dim_name} Sub-category Risk Breakdown — {entity}',
            font    = dict(size=13, color=dim_colour),
        ),
        xaxis       = dict(title='Mean Risk Z-score', zeroline=False, range=[-2.2, 2.6]),
        yaxis       = dict(title='', tickfont=dict(size=11)),
        margin      = dict(l=10, r=60, t=48, b=36),
        plot_bgcolor= 'white',
        paper_bgcolor='white',
        showlegend  = False,
    )
    return fig


def make_category_heatmap(entity, cat_data, selected_cats=None):
    """
    Heatmap: rows = sub-categories (E + S merged), column = [E mean_risk_z, S mean_risk_z,
    E high_risk%, S high_risk%].  Gives a side-by-side overview of both dimensions.
    """
    import pandas as pd

    rows = []
    for dim_name in ['Environmental', 'Social']:
        df = cat_data.get(dim_name, pd.DataFrame())
        if df.empty:
            continue
        df_ent = df[df['company'] == entity]
        for _, r in df_ent.iterrows():
            rows.append({
                'Dimension': dim_name[:3],    # 'Env' / 'Soc'
                'Category' : r['category'],
                'mean_z'   : r['mean_risk_z'],
                'hi_pct'   : r['high_risk_pct'] * 100,
                'n'        : int(r['n_metrics']),
            })

    if not rows:
        return go.Figure()

    df_all = pd.DataFrame(rows)
    if selected_cats:
        df_all = df_all[df_all['Category'].isin(selected_cats)]
    if df_all.empty:
        return go.Figure()

    # Pivot: rows = Category+Dim, cols = [Mean Z, High-Risk %]
    df_all['Label'] = df_all['Dimension'] + ' | ' + df_all['Category']
    df_all = df_all.sort_values(['Dimension', 'mean_z'], ascending=[True, False])

    z_matrix   = [[row['mean_z'],   row['hi_pct']]   for _, row in df_all.iterrows()]
    text_matrix= [[f"{row['mean_z']:.3f}", f"{row['hi_pct']:.0f}%"] for _, row in df_all.iterrows()]

    fig = go.Figure(go.Heatmap(
        z           = z_matrix,
        x           = ['Mean Risk Z-score', 'High-Risk Metrics (%)'],
        y           = df_all['Label'].tolist(),
        text        = text_matrix,
        texttemplate= '%{text}',
        textfont    = dict(size=11),
        colorscale  = [
            [0.00, '#0ea5e9'],
            [0.30, '#16a34a'],
            [0.55, '#d97706'],
            [1.00, '#dc2626'],
        ],
        colorbar    = dict(title='Risk Level', thickness=12, len=0.8),
        zmid        = 0,
    ))
    fig.update_layout(
        height       = max(300, 38 * len(df_all)),
        title        = dict(
            text     = f'ESG Category Heatmap — {entity}',
            font     = dict(size=13, color='#1e3a8a'),
        ),
        xaxis        = dict(side='top', tickfont=dict(size=11)),
        yaxis        = dict(tickfont=dict(size=10), autorange='reversed'),
        margin       = dict(l=10, r=10, t=70, b=20),
        plot_bgcolor = 'white',
        paper_bgcolor= 'white',
    )
    return fig


def make_metric_detail_chart(entity, cat_data, dim_name, category):
    """
    Horizontal bar of individual metric risk_z values within one sub-category.
    Only shown when user drills into a specific category.
    """
    import pandas as pd

    df = cat_data.get(dim_name, pd.DataFrame())
    if df.empty:
        return go.Figure()

    # We need the normalised frame, so we reconstruct it from the stored agg
    # This function receives cat_data which only has aggregated stats.
    # We show the category stats as a single row in this case.
    df_ent = df[(df['company'] == entity) & (df['category'] == category)]
    if df_ent.empty:
        return go.Figure()

    # Build a simple summary bar since we only have aggregated data here
    row   = df_ent.iloc[0]
    cats  = ['Mean Risk Z', 'High-Risk %']
    vals  = [row['mean_risk_z'], row['high_risk_pct']]
    cols  = ['#2563eb', '#dc2626']

    fig = go.Figure(go.Bar(
        x=cats, y=vals, marker_color=cols,
        text=[f"{vals[0]:.4f}", f"{vals[1]*100:.1f}%"],
        textposition='outside',
    ))
    fig.update_layout(
        height=240,
        title=dict(text=f'{category} — {dim_name} detail', font=dict(size=12)),
        yaxis=dict(range=[-0.1, 1.1]),
        margin=dict(l=20, r=20, t=44, b=30),
        plot_bgcolor='white', paper_bgcolor='white',
    )
    return fig


# =========================================================================
# Cached FL pipeline — runs once per Streamlit session
# =========================================================================

if _HAVE_DASHBOARD:
    @st.cache_resource(show_spinner=False)
    def run_fl_cached():
        """Run FL pipeline on both scenarios. Cached once per session."""
        try:
            _dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            _dir = os.getcwd()

        fp_base = os.path.join(_dir, 'ESG_Raw_Data_Supply_Chain.xlsx')
        fp_risk = os.path.join(_dir, 'ESG_Raw_Data_Supply_Chain - Risk Added.xlsx')

        res_base = run_fl_pipeline(fp_base, 'BASELINE',   verbose=False)
        res_risk = run_fl_pipeline(fp_risk, 'RISK ADDED', verbose=False)
        return res_base, res_risk
    @st.cache_resource(show_spinner=False)
    def load_category_breakdown_cached():
        """
        Pre-compute per-company, per-category breakdown for both scenarios.
        Cached once per session — reuses FL core data loading functions.
        Returns dict: {'Baseline': {dim: df}, 'Risk Added': {dim: df}}
        """
        try:
            _dir = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            _dir = os.getcwd()

        fp_base = os.path.join(_dir, 'ESG_Raw_Data_Supply_Chain.xlsx')
        fp_risk = os.path.join(_dir, 'ESG_Raw_Data_Supply_Chain - Risk Added.xlsx')

        return {
            'Baseline':   compute_category_breakdown(fp_base),
            'Risk Added': compute_category_breakdown(fp_risk),
        }

else:
    def run_fl_cached():
        raise RuntimeError('Streamlit / Plotly not installed.')

    def load_category_breakdown_cached():
        raise RuntimeError('Streamlit / Plotly not installed.')


# =========================================================================
# DASHBOARD MAIN
# =========================================================================

def dashboard_main():
    """Streamlit application for FL ESG Risk Dashboard."""
    if not _HAVE_DASHBOARD:
        print('Dashboard requires: pip install streamlit plotly')
        return

    # ---- Page config (must be the very first Streamlit call) ----
    st.set_page_config(
        page_title='FL ESG Risk Dashboard',
        page_icon='🌿',
        layout='wide',
        initial_sidebar_state='expanded',
    )
    st.markdown(DASH_CSS, unsafe_allow_html=True)

    # =========================================================
    # SIDEBAR
    # =========================================================

    # ---- Identity ----
    st.sidebar.markdown('## 🏢 My Identity')
    current_entity = st.sidebar.selectbox(
        'Select your company',
        options=ENTITY_ORDER,
        index=4,  # default: Mattel (Focal)
        help='Your dashboard view is restricted to your direct supply chain partners.',
    )

    # ---- Page navigation ----
    page = st.sidebar.radio(
        '📑 View',
        ['📊 Risk Dashboard', 'ℹ️ FL System Info'],
        index=0,
        key='page_nav',
        help='Switch between the main risk dashboard and FL system information & privacy model.',
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
            'OFF (default): Risk sliders are initialised from FL-computed values.\n'
            'ON: Sliders start at 0.500 for fresh manual self-assessment '
            '(simulates live/real-time data entry independent of the FL pipeline).'
        ),
        key='live_mode_toggle',
    )
    live_auto = False
    live_interval_sec = 2.0
    live_drift_sigma = 0.012

    if live_mode:
        live_auto = st.sidebar.toggle(
            '🔁 Auto Live Drift',
            value=True,
            help='Auto-refresh and apply small random drift for all entities (your entity remains manually adjustable).',
            key='live_auto_toggle',
        )
        live_interval_sec = st.sidebar.slider(
            'Refresh interval (seconds)',
            min_value=1.0,
            max_value=10.0,
            value=2.0,
            step=0.5,
            key='live_interval_sec',
        )
        live_drift_sigma = st.sidebar.slider(
            'Drift amplitude (σ)',
            min_value=0.001,
            max_value=0.050,
            value=0.012,
            step=0.001,
            key='live_drift_sigma',
            help='Higher value = more volatile live movement.',
        )

    if live_mode:
        st.sidebar.markdown(
            "<div style='background:#fef3c7;border:1.5px solid #f59e0b;"
            "border-radius:8px;padding:0.5rem 0.8rem;color:#78350f;"
            "font-size:0.85rem;'>"
            "🔴 <b>Live Override Active</b><br>"
            "Sliders start at 0.500. Drag to set your real-time self-reported risk."
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.sidebar.markdown(
            "<div style='background:#f0fdf4;border:1.5px solid #86efac;"
            "border-radius:8px;padding:0.5rem 0.8rem;color:#166534;"
            "font-size:0.85rem;'>"
            "🟢 <b>FL Data Mode (Default)</b><br>"
            "Sliders initialised from FL-computed risk scores."
            "</div>",
            unsafe_allow_html=True,
        )

    # ---- Scenario ----
    st.sidebar.markdown('---')
    st.sidebar.markdown('## 📊 Scenario')
    scenario = st.sidebar.radio(
        'Dataset',
        ['Baseline (Original Data)', 'Risk Added (Nutrien GHG ×2)'],
        index=0,
    )

    # ---- Load FL results ----
    with st.spinner('🤖 Running Federated Learning pipeline (cached after first run)…'):
        try:
            res_base, res_risk = run_fl_cached()
        except Exception as exc:
            st.error(f'FL pipeline error: {exc}')
            st.stop()

    results  = res_base if 'Baseline' in scenario else res_risk
    scenario_key = 'Baseline' if 'Baseline' in scenario else 'Risk Added'
    fl_E_loc = results.get('Environmental', {}).get('local_risks',  {})
    fl_S_loc = results.get('Social',        {}).get('local_risks',  {})
    fl_E_tot = results.get('Environmental', {}).get('total_risks',  {})
    fl_S_tot = results.get('Social',        {}).get('total_risks',  {})

    # ---- Live-mode state (auto-drift + persistent slider values) ----
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
                if ent == current_entity:
                    st.session_state[e_key] = 0.500
                    st.session_state[s_key] = 0.500
                else:
                    st.session_state[e_key] = base_e
                    st.session_state[s_key] = base_s
            else:
                if e_key not in st.session_state:
                    st.session_state[e_key] = base_e
                if s_key not in st.session_state:
                    st.session_state[s_key] = base_s

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
                            st.session_state[e_key] + np.random.normal(0, live_drift_sigma), 0.0, 1.0
                        ))
                        st.session_state[s_key] = float(np.clip(
                            st.session_state[s_key] + np.random.normal(0, live_drift_sigma), 0.0, 1.0
                        ))
                st.session_state[live_t_key] = now_ts

    st.session_state['prev_live_mode'] = live_mode

    # ---- My own risk score sliders ----
    st.sidebar.markdown('---')
    st.sidebar.markdown('## 📐 My Risk Scores')
    st.sidebar.markdown(
        "<div style='background:#eff6ff;border:1px solid #bfdbfe;border-radius:6px;"
        "padding:0.45rem 0.75rem;font-size:0.82rem;color:#1e40af;margin-bottom:0.4rem;'>"
        "<b>💡 FL Baseline vs Self-Assessment</b><br>"
        "• <b>FL Baseline</b>: Model-based estimate from historical ESG data<br>"
        "• <b>Self-Assessment</b>: User-adjusted score based on real-time information"
        "</div>",
        unsafe_allow_html=True,
    )

    fl_E_hint = round(float(fl_E_loc.get(current_entity, 0.5)), 4)
    fl_S_hint = round(float(fl_S_loc.get(current_entity, 0.5)), 4)
    default_E = round(float(fl_E_loc.get(current_entity, 0.5)), 4) if not live_mode else 0.500
    default_S = round(float(fl_S_loc.get(current_entity, 0.5)), 4) if not live_mode else 0.500

    if live_mode:
        my_E_local = st.sidebar.slider(
            '🔴 Environmental Risk (Local)',
            min_value=0.000, max_value=1.000,
            step=0.001,
            key=live_e_key,
            help=f'FL-computed value: {fl_E_hint:.4f} | Live mode: auto drift + manual override',
        )
        my_S_local = st.sidebar.slider(
            '🔴 Social Risk (Local)',
            min_value=0.000, max_value=1.000,
            step=0.001,
            key=live_s_key,
            help=f'FL-computed value: {fl_S_hint:.4f} | Live mode: auto drift + manual override',
        )
        st.sidebar.caption(
            f'FL baseline (reference): E={fl_E_hint:.4f}, S={fl_S_hint:.4f}  |  '
            f'Auto={("ON" if live_auto else "OFF")}  |  '
            f'Interval={live_interval_sec:.1f}s  |  σ={live_drift_sigma:.3f}'
        )
    else:
        my_E_local = st.sidebar.slider(
            'Environmental Risk (Local)',
            min_value=0.000, max_value=1.000,
            value=default_E, step=0.001,
            help=f'FL baseline: {fl_E_hint:.4f}',
            key=f'slider_my_E_fl_{current_entity}_{scenario_key}',
        )
        my_S_local = st.sidebar.slider(
            'Social Risk (Local)',
            min_value=0.000, max_value=1.000,
            value=default_S, step=0.001,
            help=f'FL baseline: {fl_S_hint:.4f}',
            key=f'slider_my_S_fl_{current_entity}_{scenario_key}',
        )
        _e_overridden = abs(my_E_local - fl_E_hint) > 0.005
        _s_overridden = abs(my_S_local - fl_S_hint) > 0.005
        _override_tag = '⚠️ Overridden — ' if (_e_overridden or _s_overridden) else ''
        st.sidebar.caption(
            f'{_override_tag}FL baseline: E={fl_E_hint:.4f}, S={fl_S_hint:.4f}'
        )

    # ---- Propagation weights — Procurement % interface ----
    st.sidebar.markdown('---')
    st.sidebar.markdown('## ⚖️ Supply Chain Dependency Weights')
    st.sidebar.caption(
        'Enter procurement share (%) for each party. '
        'The system derives the internal risk weights automatically.'
    )

    pw        = deepcopy(PROPAGATION_WEIGHTS)
    default_w = PROPAGATION_WEIGHTS[current_entity]

    if not upstream_ents:
        # No upstream — own operations = 100 %, β = 1.0
        st.sidebar.info(
            f'{current_entity} has no upstream suppliers.  \n'
            'Own operations = **100%** → β = 1.00'
        )
        my_beta  = 1.0
        my_alpha = {}

    elif len(upstream_ents) == 1:
        u0 = upstream_ents[0]
        default_u0_pct = round(float(default_w['alpha'].get(u0, 0.5)) * 100)
        default_own_pct = 100 - default_u0_pct

        u0_pct = st.sidebar.slider(
            f'📦 {u0} procurement share (%)',
            min_value=0, max_value=100,
            value=default_u0_pct, step=1,
            key=f'pct_{u0}_{current_entity}',
        )
        own_pct = 100 - u0_pct
        st.sidebar.caption(f'Own operations: **{own_pct}%** (auto)')

        my_beta  = round(own_pct / 100, 4)
        my_alpha = {u0: round(u0_pct / 100, 4)}
        _pct_map   = {u0: u0_pct}
        _own_label = own_pct

    elif len(upstream_ents) == 2:
        u0, u1 = upstream_ents[0], upstream_ents[1]
        default_u0_pct  = round(float(default_w['alpha'].get(u0, 0.12)) * 100)
        default_u1_pct  = round(float(default_w['alpha'].get(u1, 0.18)) * 100)
        default_own_pct = 100 - default_u0_pct - default_u1_pct

        u0_pct = st.sidebar.slider(
            f'📦 {u0} procurement share (%)',
            min_value=0, max_value=100,
            value=default_u0_pct, step=1,
            key=f'pct_{u0}_{current_entity}',
        )
        u1_pct = st.sidebar.slider(
            f'📦 {u1} procurement share (%)',
            min_value=0, max_value=100,
            value=default_u1_pct, step=1,
            key=f'pct_{u1}_{current_entity}',
        )
        own_pct  = 100 - u0_pct - u1_pct
        _pct_sum = u0_pct + u1_pct + max(own_pct, 0)

        # Real-time sum indicator
        if own_pct < 0:
            st.sidebar.markdown(
                f"<div style='background:#fef2f2;border:1px solid #fca5a5;border-radius:6px;"
                f"padding:0.35rem 0.6rem;font-size:0.82rem;color:#991b1b;'>"
                f"⚠️ Total exceeds 100% by {-own_pct}% — reduce supplier shares."
                f"</div>", unsafe_allow_html=True,
            )
            own_pct = 0
        else:
            st.sidebar.caption(f'Own operations: **{own_pct}%** (auto = 100 − {u0_pct} − {u1_pct})')

        my_beta  = round(own_pct / 100, 4)
        my_alpha = {u0: round(u0_pct / 100, 4), u1: round(u1_pct / 100, 4)}
        _pct_map   = {u0: u0_pct, u1: u1_pct}
        _own_label = own_pct

    else:
        # 3+ upstream — equal split fallback
        n = len(upstream_ents)
        per_pct = 100 // (n + 1)
        own_pct = 100 - per_pct * n
        my_beta  = round(own_pct / 100, 4)
        my_alpha = {u: round(per_pct / 100, 4) for u in upstream_ents}
        _pct_map   = {u: per_pct for u in upstream_ents}
        _own_label = own_pct

    pw[current_entity] = {'beta': my_beta, 'alpha': my_alpha}

    # ── Confirmation box ──────────────────────────────────────────────────
    w_sum = my_beta + sum(my_alpha.values())
    if upstream_ents and abs(w_sum - 1.0) < 0.01:
        _conf_lines = []
        for _u, _a in my_alpha.items():
            _conf_lines.append(
                f"&nbsp;&nbsp;{_u}: <b>{round(_a*100)}%</b> → α = {_a:.2f}"
            )
        _conf_lines.append(
            f"&nbsp;&nbsp;Own operations: <b>{round(my_beta*100)}%</b> → β = {my_beta:.2f}"
        )
        st.sidebar.markdown(
            "<div style='background:#f0fdf4;border:1px solid #86efac;border-radius:6px;"
            "padding:0.45rem 0.75rem;font-size:0.81rem;color:#166534;margin-top:0.4rem;'>"
            "✅ <b>Weight configuration confirmed</b><br>"
            + "<br>".join(_conf_lines)
            + "<br><span style='color:#9ca3af;font-size:0.75rem;'>"
            "Auditor-confirmed method: W1 (Purchase Value Weight)</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    elif upstream_ents:
        st.sidebar.markdown(
            f"<div style='background:#fef2f2;border:1px solid #fca5a5;border-radius:6px;"
            f"padding:0.35rem 0.6rem;font-size:0.82rem;color:#991b1b;'>"
            f"⚠️ Shares sum to {round(w_sum*100)}% — must equal 100%."
            f"</div>", unsafe_allow_html=True,
        )

    # =========================================================
    # COMPUTE PROPAGATED RISKS WITH CUSTOM WEIGHTS
    # =========================================================

    # FL mode: only current entity is adjusted by sliders.
    # Live mode: all entities use live session-state values (auto drift + current manual override).
    if live_mode:
        adj_E_loc = {
            ent: float(st.session_state.get(f'live_e_state_{ent}_{scenario_key}', fl_E_loc.get(ent, 0.5)))
            for ent in ENTITY_ORDER
        }
        adj_S_loc = {
            ent: float(st.session_state.get(f'live_s_state_{ent}_{scenario_key}', fl_S_loc.get(ent, 0.5)))
            for ent in ENTITY_ORDER
        }
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

    # =========================================================
    # MAIN CONTENT  (conditional on page selection)
    # =========================================================

    updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # =========================================================
    # PAGE: FL SYSTEM INFO  (shown when user selects it from sidebar)
    # =========================================================
    if page == 'ℹ️ FL System Info':
        st.markdown(
            "<div class='dash-header'>ℹ️ FL System Information & Privacy Model</div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='dash-sub'>"
            f"Federated Learning architecture, supply chain structure, "
            f"active weights, and privacy guarantees for <b>{display_name}</b> ({tier})"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.caption(f"Viewing as: {display_name} | {tier} | {role} | Updated: {updated}")
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
                    f"- **{DISPLAY_NAMES.get(ent, ent)}** ({cfg['tier']}) &nbsp; "
                    f"upstream: {up}{you}{vis}"
                )
            st.markdown('')
            st.markdown('### 👁️ Visibility Rules')
            st.markdown(
                f"As **{display_name}** ({tier}), you can see:  \n"
                + ',  \n'.join([f"  • {DISPLAY_NAMES.get(e, e)}" for e in visible_ents])
            )
            st.markdown('')
            st.markdown('### 🔗 Propagation Chain')
            st.markdown(
                "Risk flows **upstream → downstream**:\n\n"
                "```\n"
                "Nutrien(T2)  ──┐\n"
                "               ├──► Goodyear(T1) ──┐\n"
                "Deere(T2)    ──┘                   ├──► Mattel(Focal) ──┬──► Walmart(Down)\n"
                "                  Sherwin-W(T1)  ──┘                   └──► Target(Down)\n"
                "```"
            )

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
                "<li>Only <b>SHA-256 hashed</b> model parameters are transmitted</li>"
                f"<li>FedAvg aggregation over <b>{FL_ROUNDS} communication rounds</b></li>"
                "<li>Your risk score is <b>invisible</b> to all other entities</li>"
                "<li>Other entities' risk scores are <b>invisible</b> to you</li>"
                "</ul>"
                "</div>",
                unsafe_allow_html=True,
            )

            st.markdown('')
            st.markdown('### 🔄 Data Mode Explanation')
            st.markdown(
                "- 🟢 **FL Data Mode**: Sliders initialised from FL-computed values "
                "(reproducible, based on historical ESG data)\n"
                "- 🔴 **Live Override Mode**: Sliders start at 0.500, "
                "representing fresh real-time self-assessment input "
                "independent of the FL pipeline"
            )

            st.markdown('')
            st.markdown('### 🤖 FL Algorithm Details')
            import pandas as _pd_sys
            _sys_rows = [
                {'Parameter': 'Algorithm',            'Value': 'FedAvg (Federated Averaging)'},
                {'Parameter': 'Communication Rounds', 'Value': str(FL_ROUNDS)},
                {'Parameter': 'Local Model',          'Value': 'Logistic Regression (z-score normalised)'},
                {'Parameter': 'Privacy Mechanism',    'Value': 'SHA-256 parameter hashing (simulation)'},
                {'Parameter': 'ESG Dimensions',       'Value': 'Environmental, Social (Governance excluded)'},
                {'Parameter': 'Risk Propagation',     'Value': 'α_ij = w_ij (no decay function)'},
                {'Parameter': 'Participating Firms',  'Value': str(len(SUPPLY_CHAIN))},
            ]
            st.dataframe(_pd_sys.DataFrame(_sys_rows), hide_index=True, use_container_width=True)

        st.markdown('---')
        st.caption(
            f"FL ESG Risk Dashboard  |  {display_name} ({tier})  |  "
            f"FedAvg {FL_ROUNDS} rounds  |  SHA-256 privacy  |  {updated}"
        )
        return  # stop here — do not render the main dashboard below

    # =========================================================
    # PAGE: RISK DASHBOARD  (default page)
    # =========================================================

    # ---- Header ----
    st.markdown(
        f"<div class='dash-header'>🌿 FL ESG Risk Dashboard — {display_name}</div>",
        unsafe_allow_html=True,
    )
    mode_badge = (
        "<span style='background:#ef4444;color:white;border-radius:6px;"
        "padding:2px 8px;font-size:0.85rem;'>🔴 LIVE OVERRIDE</span>"
        if live_mode else
        "<span style='background:#16a34a;color:white;border-radius:6px;"
        "padding:2px 8px;font-size:0.85rem;'>🟢 FL DATA</span>"
    )
    st.markdown(
        f"<div class='dash-sub'>"
        f"{tier} &nbsp;|&nbsp; {role} &nbsp;|&nbsp; "
        f"Federated Learning Supply Chain Risk Management &nbsp;|&nbsp; "
        f"{scenario} &nbsp;|&nbsp; {mode_badge}"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Visible network: {len(visible_ents)} entities  |  "
        f"Updated: {updated}  |  FL rounds: {FL_ROUNDS}  |  "
        f"Algorithm: FedAvg  |  Privacy: SHA-256 simulation"
    )

    # ---- Live mode banner ----
    if live_mode:
        st.markdown(
            "<div class='live-banner'>"
            "🔴 <b>Live Override Mode Active</b> — "
            "Risk scores are manually set (not from FL pipeline). "
            "Use the sidebar sliders to enter your real-time self-reported E and S scores. "
            "Toggle off to return to FL-computed baseline values."
            "</div>",
            unsafe_allow_html=True,
        )

    # ── Section 1: My Risk Score ──────────────────────────────────────────
    st.markdown('---')
    st.subheader('📊 My Risk Assessment')
    st.markdown(
        "<div class='privacy-badge'>🔒 <b>Privacy:</b> Only YOUR own risk scores are "
        "displayed below. All other supply chain partners' scores remain confidential "
        "under the federated privacy model.</div>",
        unsafe_allow_html=True,
    )
    st.markdown('')

    # Delta from FL baseline (for comparison in live mode)
    fl_E_tot_me = float(fl_E_tot.get(current_entity, 0.5))
    fl_S_tot_me = float(fl_S_tot.get(current_entity, 0.5))
    fl_comp_me  = 0.5 * fl_E_tot_me + 0.5 * fl_S_tot_me

    # Summary cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"<div class='{'card-live' if live_mode else _card_class(my_composite)}'>"
            f"<b>Composite ESG Risk</b><br>"
            f"<span style='font-size:1.9rem'><b>{my_composite:.4f}</b></span><br>"
            f"<span class='tiny'>{lbl_C}"
            + (f" &nbsp;|&nbsp; FL={fl_comp_me:.4f}" if live_mode else "") +
            f"</span></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div class='{'card-live' if live_mode else _card_class(my_E_total)}'>"
            f"<b>Environmental (Total)</b><br>"
            f"<span style='font-size:1.9rem'><b>{my_E_total:.4f}</b></span><br>"
            f"<span class='tiny'>{lbl_E} &nbsp;|&nbsp; local={my_E_local:.4f}"
            + (f"<br>FL total={fl_E_tot_me:.4f}" if live_mode else "") +
            f"</span></div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='{'card-live' if live_mode else _card_class(my_S_total)}'>"
            f"<b>Social (Total)</b><br>"
            f"<span style='font-size:1.9rem'><b>{my_S_total:.4f}</b></span><br>"
            f"<span class='tiny'>{lbl_S} &nbsp;|&nbsp; local={my_S_local:.4f}"
            + (f"<br>FL total={fl_S_tot_me:.4f}" if live_mode else "") +
            f"</span></div>",
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"<div class='card-blue'>"
            f"<b>FL System</b><br>"
            f"<span style='font-size:1.6rem'>{FL_ROUNDS} rounds</span><br>"
            f"<span class='tiny'>FedAvg &nbsp;|&nbsp; SHA-256 privacy</span></div>",
            unsafe_allow_html=True,
        )

    # Gauge charts
    g1, g2, g3 = st.columns(3)
    with g1:
        st.plotly_chart(
            make_gauge(my_E_total, 'Environmental Risk (Total)', height=265),
            use_container_width=True, key=f'gauge_E_{current_entity}',
        )
        st.markdown(
            "<div style='display:flex;justify-content:space-between;"
            "margin-top:-14px;padding:0 8px 4px;'>"
            "<span style='color:#16a34a;font-weight:700;font-size:13px;'>Low</span>"
            "<span style='color:#dc2626;font-weight:700;font-size:13px;'>High</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    with g2:
        st.plotly_chart(
            make_gauge(my_S_total, 'Social Risk (Total)', height=265),
            use_container_width=True, key=f'gauge_S_{current_entity}',
        )
        st.markdown(
            "<div style='display:flex;justify-content:space-between;"
            "margin-top:-14px;padding:0 8px 4px;'>"
            "<span style='color:#16a34a;font-weight:700;font-size:13px;'>Low</span>"
            "<span style='color:#dc2626;font-weight:700;font-size:13px;'>High</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    with g3:
        st.plotly_chart(
            make_gauge(my_composite, 'Composite ESG Risk', height=265),
            use_container_width=True, key=f'gauge_C_{current_entity}',
        )
        st.markdown(
            "<div style='display:flex;justify-content:space-between;"
            "margin-top:-14px;padding:0 8px 4px;'>"
            "<span style='color:#16a34a;font-weight:700;font-size:13px;'>Low</span>"
            "<span style='color:#dc2626;font-weight:700;font-size:13px;'>High</span>"
            "</div>",
            unsafe_allow_html=True,
        )

    # ── FIX 1: Propagation formula → readable table ───────────────────────
    st.markdown('**📐 How is my Environmental Risk (Total) calculated?**')
    st.caption(
        'Your **total** risk score is not just your own — it also absorbs risk from '
        'upstream suppliers. The table below shows exactly how each component contributes:'
    )
    w_cur = pw[current_entity]
    _formula_rows = []
    _pct_denom = my_E_total if abs(my_E_total) > 1e-9 else 1.0
    _own_contrib = w_cur['beta'] * my_E_local
    _formula_rows.append({
        'Component':    'Your own risk  (β × E_local)',
        'Weight':       f"β = {w_cur['beta']:.4f}",
        'Risk Score':   f"E_local = {my_E_local:.4f}",
        'Contribution': f"{_own_contrib:.4f}",
        'Share':        f"{_own_contrib / _pct_denom * 100:.1f}%",
    })
    for _s, _a in w_cur['alpha'].items():
        _r_s    = float(tot_E.get(_s, 0.5))
        _contrib = _a * _r_s
        _formula_rows.append({
            'Component':    f'Upstream: {_s}  (α × E_total)',
            'Weight':       f"α = {_a:.4f}",
            'Risk Score':   f"E_total({_s}) = {_r_s:.4f}",
            'Contribution': f"{_contrib:.4f}",
            'Share':        f"{_contrib / _pct_denom * 100:.1f}%",
        })
    _formula_rows.append({
        'Component':    '➜  E_total  (sum of all rows)',
        'Weight':       '—',
        'Risk Score':   '—',
        'Contribution': f"{my_E_total:.4f}",
        'Share':        '100%',
    })
    import pandas as _pd_local
    st.dataframe(_pd_local.DataFrame(_formula_rows), hide_index=True, use_container_width=True)
    st.caption(
        '**How to read this:** E_total = (β × your own E_local) + Σ(α_j × upstream j\'s E_total).  '
        'Use the ⚖️ sidebar sliders to change β and α and watch the contributions shift.'
    )

    # ── Section 2: Risk Mitigation Recommendations ───────────────────────
    st.markdown('---')
    st.subheader('🎯 Risk Mitigation Recommendations')
    st.caption(
        'Rule-based recommendations generated from your FL risk scores, '
        'category breakdown, tier position, and upstream propagation weights. '
        'Each recommendation includes a relevant regulatory reference.'
    )

    # Load category data for recommendations
    try:
        _cat_all_recs = load_category_breakdown_cached()
        _cat_data_recs = _cat_all_recs.get(scenario_key, {})
    except Exception:
        _cat_data_recs = {}

    if live_mode:
        _cat_data_recs = build_live_category_view(
            _cat_data_recs,
            entity=current_entity,
            base_E_local=fl_E_hint,
            base_S_local=fl_S_hint,
            cur_E_local=my_E_local,
            cur_S_local=my_S_local,
        )

    recs = generate_recommendations(
        entity       = current_entity,
        tier         = tier,
        role         = role,
        E_local      = my_E_local,
        S_local      = my_S_local,
        E_total      = my_E_total,
        S_total      = my_S_total,
        composite    = my_composite,
        cat_data     = _cat_data_recs,
        pw           = pw,
        scenario_key = scenario_key,
        upstream_ents= upstream_ents,
    )

    # ---- Priority filter ----
    priority_filter = st.multiselect(
        'Filter by priority',
        options=['HIGH', 'MEDIUM', 'LOW', 'INFO'],
        default=['HIGH', 'MEDIUM', 'LOW', 'INFO'],
        key=f'rec_priority_filter_{current_entity}',
    )

    recs_filtered = [r for r in recs if r['priority'] in priority_filter]

    # ---- Summary counts ----
    from collections import Counter
    counts = Counter(r['priority'] for r in recs)
    rc1, rc2, rc3, rc4 = st.columns(4)
    for col, pri, emoji in zip(
        [rc1, rc2, rc3, rc4],
        ['HIGH', 'MEDIUM', 'LOW', 'INFO'],
        ['🔴', '🟡', '🟢', '🔵'],
    ):
        with col:
            st.metric(label=f'{emoji} {pri}', value=counts.get(pri, 0))

    st.markdown('')

    if not recs_filtered:
        st.info('No recommendations match the selected priority filter.')
    else:
        for rec in recs_filtered:
            st.markdown(render_recommendation_card(rec), unsafe_allow_html=True)

    # ── Section 3: Weights Visualisation ─────────────────────────────────
    st.markdown('---')
    st.subheader('⚖️ My Propagation Weights')

    wv_col1, wv_col2 = st.columns([1, 1])
    with wv_col1:
        st.plotly_chart(
            make_weight_radar(current_entity, pw),
            use_container_width=True,
            key=f'weight_radar_{current_entity}',
        )
    with wv_col2:
        st.markdown('**Active Weight Configuration**')
        w = pw[current_entity]
        weight_rows = [{'Component': f'β ({current_entity})', 'Weight': f'{w["beta"]:.4f}', 'Meaning': 'Own local risk'}]
        for s, a in w['alpha'].items():
            weight_rows.append({'Component': f'α ({s})', 'Weight': f'{a:.4f}', 'Meaning': f'Upstream: {s}'})
        total_w = w['beta'] + sum(w['alpha'].values())
        weight_rows.append({'Component': 'SUM', 'Weight': f'{total_w:.4f}', 'Meaning': '✅ Should equal 1.00'})

        import pandas as pd
        st.dataframe(pd.DataFrame(weight_rows), hide_index=True, use_container_width=True)

        st.markdown('')
        st.markdown(
            f"**Formula:**  R_total = β × R_local + Σ α_j × R_total(upstream_j)  \n"
            f"  β = {w['beta']:.4f}  |  "
            + '  |  '.join([f"α({s}) = {a:.4f}" for s, a in w['alpha'].items()])
            if w['alpha'] else f"**Formula:**  R_total = β × R_local = {w['beta']:.4f} × R_local  (no upstream)"
        )

    # ── Section 3: Category Risk Breakdown ───────────────────────────────
    st.markdown('---')
    st.subheader('📋 Category Risk Breakdown')
    if live_mode:
        st.caption(
            'Live mode view: sub-category scores are dynamically adjusted from your '
            'current local E/S inputs (simulated real-time mapping). '
            'Higher z-score = higher relative risk.'
        )
    else:
        st.caption(
            'Per sub-category mean risk z-score for **your company only** '
            '(pure visualisation — does not change R_local or R_total). '
            'Higher z-score = higher risk relative to supply chain peers.'
        )

    # Load category breakdown (cached)
    try:
        cat_breakdown_all = load_category_breakdown_cached()
    except Exception as _exc:
        st.warning(f'Category breakdown unavailable: {_exc}')
        cat_breakdown_all = {}

    cat_data = cat_breakdown_all.get(scenario_key, {})
    cat_data_view = (
        build_live_category_view(
            cat_data,
            entity=current_entity,
            base_E_local=fl_E_hint,
            base_S_local=fl_S_hint,
            cur_E_local=my_E_local,
            cur_S_local=my_S_local,
        ) if live_mode else cat_data
    )

    # ---- Category filter (multi-select) ----
    all_cats_E = []
    all_cats_S = []
    if cat_data_view:
        import pandas as pd
        df_e = cat_data_view.get('Environmental', pd.DataFrame())
        df_s = cat_data_view.get('Social',        pd.DataFrame())
        if not df_e.empty:
            all_cats_E = sorted(df_e['category'].unique().tolist())
        if not df_s.empty:
            all_cats_S = sorted(df_s['category'].unique().tolist())

    filt_col1, filt_col2 = st.columns(2)
    with filt_col1:
        sel_cats_E = st.multiselect(
            '🌱 Environmental sub-categories to highlight',
            options=all_cats_E,
            default=all_cats_E,
            key=f'cat_filter_E_{current_entity}',
            help='Deselect to grey-out categories. Calculation is unchanged.',
        )
    with filt_col2:
        sel_cats_S = st.multiselect(
            '👥 Social sub-categories to highlight',
            options=all_cats_S,
            default=all_cats_S,
            key=f'cat_filter_S_{current_entity}',
            help='Deselect to grey-out categories. Calculation is unchanged.',
        )

    # ---- View selector ----
    view_mode = st.radio(
        'Display style',
        ['Side-by-side bars (E & S)', 'Combined heatmap'],
        horizontal=True,
        key=f'cat_view_{current_entity}',
    )

    if not cat_data_view:
        st.info('Category breakdown data not available.')
    elif view_mode == 'Side-by-side bars (E & S)':
        cb1, cb2 = st.columns(2)
        with cb1:
            fig_e = make_category_bar_h(
                current_entity, cat_data_view, 'Environmental', sel_cats_E
            )
            if fig_e.data:
                st.plotly_chart(fig_e, use_container_width=True,
                                key=f'cat_bar_E_{current_entity}_{scenario_key}')
            else:
                st.info('No Environmental category data for this company.')
        with cb2:
            fig_s = make_category_bar_h(
                current_entity, cat_data_view, 'Social', sel_cats_S
            )
            if fig_s.data:
                st.plotly_chart(fig_s, use_container_width=True,
                                key=f'cat_bar_S_{current_entity}_{scenario_key}')
            else:
                st.info('No Social category data for this company.')
    else:
        # Heatmap view — merge E+S selected cats
        combined_sel = list(set(sel_cats_E) | set(sel_cats_S)) if (sel_cats_E or sel_cats_S) else None
        fig_hm = make_category_heatmap(current_entity, cat_data_view, combined_sel)
        if fig_hm.data:
            st.plotly_chart(fig_hm, use_container_width=True,
                            key=f'cat_hm_{current_entity}_{scenario_key}')
        else:
            st.info('No category data for this company.')

    # ---- Summary table (expandable) ----
    with st.expander('📊 Category Risk Summary Table', expanded=False):
        import pandas as pd
        rows_tbl = []
        for dim_name in ['Environmental', 'Social']:
            df_dim = cat_data_view.get(dim_name, pd.DataFrame())
            if df_dim.empty:
                continue
            df_ent = df_dim[df_dim['company'] == current_entity]
            for _, r in df_ent.iterrows():
                lbl, _ = _risk_level(max(0, min(1, (r['mean_risk_z'] + 2) / 4)))
                rows_tbl.append({
                    'Dimension'        : dim_name,
                    'Sub-Category'     : r['category'],
                    'Mean Risk Z'      : round(r['mean_risk_z'], 4),
                    'High-Risk Metrics': f"{r['high_risk_pct']*100:.1f}%",
                    '# Metrics'        : int(r['n_metrics']),
                    'Risk Level'       : lbl,
                })
        if rows_tbl:
            df_tbl = pd.DataFrame(rows_tbl)
            st.dataframe(
                df_tbl.sort_values(['Dimension', 'Mean Risk Z'], ascending=[True, False]),
                hide_index=True, use_container_width=True,
            )
            st.caption(
                'Mean Risk Z: cross-company z-score averaged over all metrics in this category. '
                'High-Risk Metrics: fraction of metrics where risk_z > 0 (above-median risk).'
            )
        else:
            st.info('No category data available.')

    # ── Section 4: Supply Chain Network ──────────────────────────────────
    st.markdown('---')
    st.subheader('🔗 Supply Chain Network')
    st.caption(
        f"Entities **visible** to {display_name}: "
        f"**{', '.join([DISPLAY_NAMES.get(e, e) for e in visible_ents])}**  "
        f"|  Grey nodes marked 🔒 = risk scores confidential"
    )
    st.plotly_chart(
        make_network(current_entity, visible_ents, tot_E, tot_S, live_mode),
        use_container_width=True,
        key=f'network_{current_entity}_{scenario[:4]}_{live_mode}',
    )

    # ── FIX 4: Connected Companies — Risk Parameters Viewer ───────────────
    _all_connected = list(set(upstream_ents + vis_info['direct_customers']))
    if _all_connected:
        st.markdown('---')
        _show_params = st.checkbox(
            '🔍 Show directly connected companies\' risk parameters',
            value=False,
            key=f'show_params_{current_entity}',
            help=(
                'Under the federated privacy model you cannot see neighbours\' risk scores, '
                'but you can view their structural β and α propagation weights.'
            ),
        )
        if _show_params:
            st.subheader('🔍 Connected Companies — FL Risk Parameters')
            st.markdown(
                "Connected companies transmit **encrypted risk parameters** to you via the FL "
                "central server — no raw operational data crosses organisational boundaries.  \n"
                "**Upstream (Tier-1 suppliers):** received parameters support CSDDD Article 7 "
                "due diligence — weighted ESG risk scores are computed locally to identify "
                "adverse supply chain impacts.  \n"
                "**Downstream (retailers):** parameters from distribution partners enable "
                "commercial ESG risk monitoring, providing early signals of reputational risk "
                "(outside CSDDD scope)."
            )
            with st.expander('📋 FL Risk Parameters table', expanded=True):
                _param_rows = []
                for _ent in ENTITY_ORDER:
                    if _ent not in _all_connected:
                        continue
                    _ent_tier = SUPPLY_CHAIN[_ent]['tier']
                    _rel = 'Supplier ↑' if _ent in upstream_ents else 'Customer ↓'
                    _e_tot = float(fl_E_tot.get(_ent, float('nan')))
                    _s_tot = float(fl_S_tot.get(_ent, float('nan')))
                    _comp  = round(0.5 * _e_tot + 0.5 * _s_tot, 4) if not (
                        _e_tot != _e_tot or _s_tot != _s_tot) else float('nan')
                    def _fmt(v):
                        return f'{v:.4f}' if v == v else '—'
                    _param_rows.append({
                        'Company':              _ent,
                        'Tier':                 _ent_tier,
                        'Relationship':         _rel,
                        'E Risk (FL)':          _fmt(_e_tot),
                        'S Risk (FL)':          _fmt(_s_tot),
                        'Composite (FL)':       _fmt(_comp),
                    })
                if _param_rows:
                    import pandas as _pd_local
                    st.dataframe(_pd_local.DataFrame(_param_rows), hide_index=True, use_container_width=True)
                    st.caption(
                        'FL risk scores are computed by the central server from aggregated '
                        'federated model outputs and shared back to directly connected participants. '
                        'Values reflect the selected scenario.'
                    )

    # ── Section 5: Risk Trend ─────────────────────────────────────────────
    st.markdown('---')
    st.subheader('📈 Risk Trend (30-day Simulated History)')
    st.caption(
        f'Trend seeded from your {"manually-set" if live_mode else "FL"} local risk scores '
        f'(E_local={my_E_local:.4f}, S_local={my_S_local:.4f}). '
        + ('Adjust the sidebar sliders to see how the trend shifts.' if not live_mode
           else '🔴 Live Override: adjust sliders to simulate different risk scenarios.')
    )
    st.plotly_chart(
        make_trend_chart(current_entity, my_E_local, my_S_local, live_mode=live_mode),
        use_container_width=True,
        key=f'trend_{current_entity}_{my_E_local:.3f}_{my_S_local:.3f}_{live_mode}',
    )

    # ── Section 6: Scenario Comparison ────────────────────────────────────
    st.markdown('---')
    st.subheader('🔄 Scenario Comparison: Baseline vs Risk Added')
    st.caption(
        'Impact of Nutrien\'s GHG emissions doubling (Tier 2) propagated to your position. '
        'Both bars use your **current sidebar weight settings** (β and α), '
        'so differences in the bars reflect the upstream data change only. '
        '⚠️ **Social (S) bars are identical by design** — the "Risk Added" scenario only '
        'modifies Nutrien\'s Environmental (GHG) data; Social local_risks are unchanged '
        'across scenarios, so S propagates to the same value in both.'
        + (' Live mode adds a "Live Now" stream.' if live_mode else '')
    )

    # ── FIX 5: Re-compute BOTH scenarios with custom weights ─────────────
    # This ensures the bar chart truly reflects the data difference between scenarios
    # (default FL weights can mask differences; custom weights show the right picture)
    _base_E_loc = res_base.get('Environmental', {}).get('local_risks', {})
    _base_S_loc = res_base.get('Social',        {}).get('local_risks', {})
    _risk_E_loc = res_risk.get('Environmental', {}).get('local_risks', {})
    _risk_S_loc = res_risk.get('Social',        {}).get('local_risks', {})

    _cust_base_E = propagate_risk_custom(_base_E_loc, pw)
    _cust_base_S = propagate_risk_custom(_base_S_loc, pw)
    _cust_risk_E = propagate_risk_custom(_risk_E_loc, pw)
    _cust_risk_S = propagate_risk_custom(_risk_S_loc, pw)

    bE = float(_cust_base_E.get(current_entity, 0.5))
    bS = float(_cust_base_S.get(current_entity, 0.5))
    rE = float(_cust_risk_E.get(current_entity, 0.5))
    rS = float(_cust_risk_S.get(current_entity, 0.5))
    bC = 0.5 * bE + 0.5 * bS
    rC = 0.5 * rE + 0.5 * rS

    if live_mode:
        disp_E, disp_S, disp_C = my_E_total, my_S_total, my_composite
    else:
        disp_E, disp_S, disp_C = rE, rS, rC

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        dE = rE - bE
        pE = (dE / bE * 100) if bE else 0
        st.metric(
            label='Environmental Risk (Live Total)' if live_mode else 'Environmental Risk (Total)',
            value=f'{disp_E:.4f}',
            delta=f'Risk-Added vs Baseline: {dE:+.4f}  ({pE:+.2f}%)',
            delta_color='inverse',
        )
        st.caption(f'Baseline: {bE:.4f}  |  Risk Added: {rE:.4f}')
    with mc2:
        dS = rS - bS
        pS = (dS / bS * 100) if bS else 0
        st.metric(
            label='Social Risk (Live Total)' if live_mode else 'Social Risk (Total)',
            value=f'{disp_S:.4f}',
            delta=f'Risk-Added vs Baseline: {dS:+.4f}  ({pS:+.2f}%)',
            delta_color='inverse',
        )
        st.caption(f'Baseline: {bS:.4f}  |  Risk Added: {rS:.4f}')
    with mc3:
        dC = rC - bC
        pC = (dC / bC * 100) if bC else 0
        st.metric(
            label='Composite Risk (Live)' if live_mode else 'Composite Risk',
            value=f'{disp_C:.4f}',
            delta=f'Risk-Added vs Baseline: {dC:+.4f}  ({pC:+.2f}%)',
            delta_color='inverse',
        )
        st.caption(f'Baseline: {bC:.4f}  |  Risk Added: {rC:.4f}')

    if live_mode:
        live_rows = [
            {'Dimension': 'Environmental', 'Baseline': round(bE, 4), 'Risk Added': round(rE, 4),
             'Live Now': round(my_E_total, 4), 'Live vs Baseline': round(my_E_total - bE, 4)},
            {'Dimension': 'Social', 'Baseline': round(bS, 4), 'Risk Added': round(rS, 4),
             'Live Now': round(my_S_total, 4), 'Live vs Baseline': round(my_S_total - bS, 4)},
            {'Dimension': 'Composite', 'Baseline': round(bC, 4), 'Risk Added': round(rC, 4),
             'Live Now': round(my_composite, 4), 'Live vs Baseline': round(my_composite - bC, 4)},
        ]
        import pandas as _pd_local
        st.dataframe(_pd_local.DataFrame(live_rows), hide_index=True, use_container_width=True)

    st.plotly_chart(
        make_scenario_bar(
            current_entity,
            res_base,
            res_risk,
            live_vals={'E': my_E_total, 'S': my_S_total, 'C': my_composite} if live_mode else None,
            precomp_base_E=bE, precomp_base_S=bS,
            precomp_risk_E=rE, precomp_risk_S=rS,
        ),
        use_container_width=True,
        key=f'scen_bar_{current_entity}',
    )

    # ── Footer ────────────────────────────────────────────────────────────
    st.caption(
        f"FL ESG Risk Dashboard  |  {display_name} ({tier})  |  "
        f"FedAvg {FL_ROUNDS} rounds  |  SHA-256 privacy  |  "
        f"{scenario}  |  {'🔴 Live Override' if live_mode else '🟢 FL Data'}  |  {updated}"
    )

    # ---- Live auto-refresh loop (optional) ----
    if live_mode and live_auto:
        time.sleep(live_interval_sec)
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()


# =============================================================================
# ENTRY POINT  — detects whether running via Streamlit or as a Python script
# =============================================================================

def _in_streamlit_ctx():
    """Return True when this script is being executed by Streamlit."""
    if not _HAVE_DASHBOARD:
        return False
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        return get_script_run_ctx() is not None
    except Exception:
        return False


if __name__ == '__main__':
    if _in_streamlit_ctx():
        # Running via:  streamlit run FL03152025_Dashboard.py
        dashboard_main()
    else:
        # Running via:  python FL03152025_Dashboard.py
        main()
