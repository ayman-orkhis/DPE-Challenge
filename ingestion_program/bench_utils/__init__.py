# A module for the benchmark, to provide utilities usable in both the
# ingestion and the submission programs.
#
# Note that the import cannot be done at the top level of
# the ingestion program

TARGET_COLUMN = "conso_5_usages_par_m2_ef"

LEAK_COLUMNS = [
    "etiquette_dpe",
    "etiquette_ges",
    "cout_chauffage",
    "cout_total_5_usages",
    "emission_ges_5_usages_par_m2",
    '_score',
    'numero_dpe',
    'type_energie_n2'
]
