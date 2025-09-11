import pandas as pd
import numpy as np
from typing import List
from collections import OrderedDict


def cohort_table(data_df:pd.DataFrame, strat_var:str, strat_var_vals:List, other_vars:List[str], patient_id_col:str=None, include_age:bool=True):
    """
    Generate a cohort characteristics table stratified by a variable.
    
    Creates demographic and statistical summary tables commonly used in medical research
    to describe study populations across different stratification groups.
    
    Args:
        data_df: DataFrame containing the cohort data
        strat_var: Column name to stratify the analysis by
        strat_var_vals: List of values in strat_var to create separate columns for
        other_vars: List of other variables to include in the table
        patient_id_col: Column name containing unique patient identifiers (optional)
        include_age: Whether to include age statistics (default True)
        
    Returns:
        pd.DataFrame: Formatted cohort characteristics table
    """
    stat_dict = OrderedDict()
    stat_dict["Variable"] = []
    stat_dict["All"] = []
    for val in strat_var_vals:
        stat_dict[val] = []
    if include_age:
        if not "age" in data_df.columns:
            try:
                data_df.rename(columns={"Age": "age"}, inplace=True)
            except:
                include_age=False
    if patient_id_col is not None:
        assert patient_id_col in data_df.columns, f"Patient ID column {patient_id_col} not in dataframe columns {data_df.columns}"
        # patients per group
        for strat_val in strat_var_vals:
            strat_df = data_df.loc[data_df[strat_var]==strat_val]
            n = strat_df[patient_id_col].nunique()
            stat_dict[strat_val].append(f"{n}")
        stat_dict["All"].append(f"{data_df[patient_id_col].nunique()}")
        stat_dict["Variable"].append("Patients")
    # records per group
    for strat_val in strat_var_vals:
        strat_df = data_df.loc[data_df[strat_var]==strat_val]
        n = len(strat_df)
        stat_dict[strat_val].append(f"{n}")
        if include_age:
            stat_dict[strat_val].append(f"{np.nanmedian(strat_df.age):.0f} ({np.nanstd(strat_df.age):.0f})")
    stat_dict["All"].append(f"{len(data_df)}")
    stat_dict["Variable"].append("Records")
    if include_age:
        stat_dict["All"].append(f"{np.nanmedian(data_df.age):.0f} ({np.nanstd(data_df.age):.0f})")
        stat_dict["Variable"].append("Median age (SD)")
    print(stat_dict)
    for ovar in other_vars:
        for strat_val in strat_var_vals:
            strat_df = data_df.loc[data_df[strat_var]==strat_val]
            value_counts = strat_df[ovar].value_counts(dropna=False)
            for val in data_df[ovar].unique():
                if val not in value_counts.keys():
                    value_counts[val] = 0
            value_count_dict = {val:value_counts[val] for val in data_df[ovar].unique()}
            for key, count in value_count_dict.items():
               stat_dict[strat_val].append(f"{count} ({count/len(strat_df)*100:.1f})")
        # ovar counts for all
        value_counts = data_df[ovar].value_counts(dropna=False)
        value_count_dict = {val:value_counts[val] for val in data_df[ovar].unique()}
        for key, count in value_count_dict.items():
           stat_dict["All"].append(f"{count} ({count/len(data_df)*100:.1f})")
        # populate info column
        for val in data_df[ovar].unique():
            stat_dict["Variable"].append(f"{val} (\%)")
    try:
        stat_df = pd.DataFrame.from_dict(stat_dict)
        return stat_df
    except Exception as e:
        print(e)
        print(f"Something went wrong:\n {stat_dict}")