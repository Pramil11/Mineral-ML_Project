"""
Parse mineral formulas into element-level features.
E.g. SiO2 → {Si:1, O:2}
Creates element count columns and weighted atomic property features.
"""
import re
import pandas as pd
import numpy as np

# Atomic masses for common elements (used for weighting)
ATOMIC_DATA = {
    'H':  {'mass': 1.008, 'en': 2.20},
    'He': {'mass': 4.003, 'en': 0.00},
    'Li': {'mass': 6.941, 'en': 0.98},
    'Be': {'mass': 9.012, 'en': 1.57},
    'B':  {'mass': 10.81, 'en': 2.04},
    'C':  {'mass': 12.01, 'en': 2.55},
    'N':  {'mass': 14.01, 'en': 3.04},
    'O':  {'mass': 16.00, 'en': 3.44},
    'F':  {'mass': 19.00, 'en': 3.98},
    'Na': {'mass': 22.99, 'en': 0.93},
    'Mg': {'mass': 24.31, 'en': 1.31},
    'Al': {'mass': 26.98, 'en': 1.61},
    'Si': {'mass': 28.09, 'en': 1.90},
    'P':  {'mass': 30.97, 'en': 2.19},
    'S':  {'mass': 32.07, 'en': 2.58},
    'Cl': {'mass': 35.45, 'en': 3.16},
    'K':  {'mass': 39.10, 'en': 0.82},
    'Ca': {'mass': 40.08, 'en': 1.00},
    'Ti': {'mass': 47.87, 'en': 1.54},
    'V':  {'mass': 50.94, 'en': 1.63},
    'Cr': {'mass': 52.00, 'en': 1.66},
    'Mn': {'mass': 54.94, 'en': 1.55},
    'Fe': {'mass': 55.85, 'en': 1.83},
    'Co': {'mass': 58.93, 'en': 1.88},
    'Ni': {'mass': 58.69, 'en': 1.91},
    'Cu': {'mass': 63.55, 'en': 1.90},
    'Zn': {'mass': 65.38, 'en': 1.65},
    'As': {'mass': 74.92, 'en': 2.18},
    'Se': {'mass': 78.96, 'en': 2.55},
    'Br': {'mass': 79.90, 'en': 2.96},
    'Sr': {'mass': 87.62, 'en': 0.95},
    'Y':  {'mass': 88.91, 'en': 1.22},
    'Zr': {'mass': 91.22, 'en': 1.33},
    'Nb': {'mass': 92.91, 'en': 1.60},
    'Mo': {'mass': 95.94, 'en': 2.16},
    'Ag': {'mass': 107.87, 'en': 1.93},
    'Sn': {'mass': 118.71, 'en': 1.96},
    'Sb': {'mass': 121.76, 'en': 2.05},
    'Te': {'mass': 127.60, 'en': 2.10},
    'Ba': {'mass': 137.33, 'en': 0.89},
    'La': {'mass': 138.91, 'en': 1.10},
    'Ce': {'mass': 140.12, 'en': 1.12},
    'Nd': {'mass': 144.24, 'en': 1.14},
    'W':  {'mass': 183.84, 'en': 2.36},
    'Pb': {'mass': 207.20, 'en': 2.33},
    'Bi': {'mass': 208.98, 'en': 2.02},
    'U':  {'mass': 238.03, 'en': 1.38},
    'Th': {'mass': 232.04, 'en': 1.30},
}

# Regex to match element + optional count (integer or decimal)
ELEMENT_PATTERN = re.compile(r'([A-Z][a-z]?)(\d*\.?\d*)')


def parse_formula(formula):
    """
    Parse a chemical formula string into a dict of {element: count}.
    Handles simple formulas like SiO2, NaAlSi3O8, etc.
    Ignores parentheses/brackets (flattens to top-level elements found).
    """
    if not isinstance(formula, str) or formula.strip() == '':
        return {}

    # Remove common non-element characters but keep element letters and numbers
    # Strip brackets, dots, middots, water notation, commas
    clean = formula.replace('·', '').replace('&middot;', '')
    
    elements = {}
    for match in ELEMENT_PATTERN.finditer(clean):
        elem = match.group(1)
        count_str = match.group(2)
        
        # Skip if not a real element symbol
        if elem not in ATOMIC_DATA:
            continue
        
        count = float(count_str) if count_str else 1.0
        elements[elem] = elements.get(elem, 0) + count
    
    return elements


def create_element_features(df, top_n=15):
    """
    Create element-level features from the clean_formula column.
    Returns the dataframe with new columns added.
    """
    # Parse all formulas
    parsed = df['clean_formula'].apply(parse_formula)
    
    # Find the most common elements across all minerals
    element_counts = {}
    for elem_dict in parsed:
        for elem in elem_dict:
            element_counts[elem] = element_counts.get(elem, 0) + 1
    
    top_elements = sorted(element_counts.keys(), key=lambda x: element_counts[x], reverse=True)[:top_n]
    print(f"Top {top_n} elements: {top_elements}")
    
    # Create count columns for top elements
    for elem in top_elements:
        df[f'element_{elem}_count'] = parsed.apply(lambda d: d.get(elem, 0))
    
    # Total unique elements per mineral
    df['total_unique_elements'] = parsed.apply(len)
    
    # Total atom count
    df['total_atom_count'] = parsed.apply(lambda d: sum(d.values()) if d else 0)
    
    # Weighted average atomic mass
    def weighted_avg_mass(elem_dict):
        if not elem_dict:
            return np.nan
        total_count = sum(elem_dict.values())
        if total_count == 0:
            return np.nan
        weighted = sum(
            ATOMIC_DATA.get(e, {}).get('mass', 0) * c
            for e, c in elem_dict.items()
        )
        return weighted / total_count
    
    # Weighted average electronegativity
    def weighted_avg_en(elem_dict):
        if not elem_dict:
            return np.nan
        total_count = sum(elem_dict.values())
        if total_count == 0:
            return np.nan
        weighted = sum(
            ATOMIC_DATA.get(e, {}).get('en', 0) * c
            for e, c in elem_dict.items()
            if ATOMIC_DATA.get(e, {}).get('en', 0) > 0  # skip noble gases etc.
        )
        return weighted / total_count
    
    df['formula_weighted_mass'] = parsed.apply(weighted_avg_mass)
    df['formula_weighted_en'] = parsed.apply(weighted_avg_en)
    
    # Store parsed dict as string for reference
    df['parsed_elements'] = parsed.apply(lambda d: str(d) if d else '{}')
    
    # List of elements as a set (for filtering in dashboard)
    df['element_list'] = parsed.apply(lambda d: ','.join(sorted(d.keys())) if d else '')
    
    return df, top_elements


if __name__ == '__main__':
    df = pd.read_csv('outputs/reports/cleaned.csv')
    print(f"Loaded {len(df)} rows")
    
    df, top_elements = create_element_features(df, top_n=15)
    
    # Summary
    print(f"\nNew columns added: {len(top_elements) + 6}")
    print(f"Sample parsed formulas:")
    for _, row in df[['name', 'clean_formula', 'parsed_elements', 'total_unique_elements']].head(10).iterrows():
        print(f"  {row['name']}: {row['clean_formula']} → {row['parsed_elements']} ({row['total_unique_elements']} elements)")
    
    df.to_csv('outputs/reports/cleaned.csv', index=False)
    print("\nSaved updated cleaned.csv")
