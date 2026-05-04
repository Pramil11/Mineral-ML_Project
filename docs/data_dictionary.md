# Data Dictionary — Mineral-ML Project

## Data Source
- **Primary Source:** [Mindat.org](https://www.mindat.org/)
- **Collection Date:** March 25, 2026
- **Method:** Manual extraction and script-based API parsing
- **Enhancement:** Chemical features computed using RDKit and Mendeleev libraries (Assignment 3b)

## Dataset Files

| File | Description | Rows |
|---|---|---|
| `data/enhanced_assignment3_dataset.csv` | Raw enhanced dataset | 3,266 |
| `outputs/reports/cleaned.csv` | Cleaned, feature-engineered dataset | ~3,168 |

## Column Descriptions

### Identifiers
| Column | Type | Description |
|---|---|---|
| `id` | int | Unique mineral ID from Mindat |
| `name` | str | Mineral name |
| `formula` | str | Original chemical formula (may contain HTML tags) |
| `clean_formula` | str | Cleaned chemical formula |

### Physical Properties
| Column | Type | Unit | Description |
|---|---|---|---|
| `hardness_min` | float | Mohs | Minimum hardness on Mohs scale |
| `hardness_max` | float | Mohs | Maximum hardness on Mohs scale |
| `density_min` | float | g/cm³ | Minimum specific gravity |
| `density_max` | float | g/cm³ | Maximum specific gravity |
| `avg_hardness` | float | Mohs | Engineered: (min + max) / 2. Valid range: [0, 10] |
| `avg_density` | float | g/cm³ | Engineered: (min + max) / 2. Valid range: [0, 25] |

### Categorical Properties
| Column | Type | Description |
|---|---|---|
| `crystal_system` | str | Crystal system. Normalized: "Isometric" → "Cubic". Values: Monoclinic, Orthorhombic, Trigonal, Triclinic, Cubic, Hexagonal, Tetragonal, Amorphous, Unknown |
| `cleavage` | str | Cleavage description |
| `transparency` | str | Transparency. Values: Transparent, Translucent, Opaque, and combinations |
| `industry_application` | str | Heuristic industry category. Values: Construction, Other, Chemical, Electronics, Energy |

### Computed Chemical Features (Mendeleev)
| Column | Type | Unit | Description |
|---|---|---|---|
| `mend_atomic_mass` | float | u (amu) | Average atomic mass of constituent elements |
| `mend_electronegativity` | float | Pauling | Average electronegativity |
| `mend_covalent_radius` | float | pm | Average covalent radius |
| `mend_ionization_energy` | float | eV | Average first ionization energy |
| `mend_melting_point` | float | K | Average melting point of elements |
| `mend_dipole_polarizability` | float | Å³ | Average dipole polarizability |

### Computed Chemical Features (RDKit)
| Column | Type | Description |
|---|---|---|
| `rdkit_exact_mw` | float | Exact molecular weight |
| `rdkit_valence_electrons` | int | Total valence electrons |
| `rdkit_heavy_atoms` | int | Number of heavy (non-hydrogen) atoms |

### Element-Level Features (from Formula Parsing)
| Column | Type | Description |
|---|---|---|
| `element_X_count` | float | Count of element X in the formula (top 15 elements) |
| `total_unique_elements` | int | Number of unique elements in the formula |
| `total_atom_count` | float | Total atom count across all elements |
| `formula_weighted_mass` | float | Weighted average atomic mass from formula |
| `formula_weighted_en` | float | Weighted average electronegativity from formula |
| `parsed_elements` | str | Dict string of parsed elements (for reference) |
| `element_list` | str | Comma-separated list of elements present |

## Cleaning Steps Applied
1. Coerce hardness/density min/max to numeric
2. Compute avg_hardness and avg_density
3. Remove outliers: hardness outside [0, 10], density outside [0, 25]
4. Fill missing categoricals with "Unknown"
5. Normalize crystal system names ("Isometric" → "Cubic")
6. Drop rows with missing/infinite avg_hardness or avg_density
7. Parse formulas into element count vectors and weighted properties
