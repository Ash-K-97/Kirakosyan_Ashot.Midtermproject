import pandas as pd
import itertools
import numpy as np
from collections import defaultdict
import pyfpgrowth
from apriori_python.apriori import apriori
import time  
print("Hello, welcome to my Midterm project")
# Create Dataset
datasetlist = ['Amazon', 'Farmmarket','Wholefood','Bestbuy','Kmart']

# Function to confirm the selected store and input min_support and min_confidence
def select_store_and_input_parameters():
    while True:
        try:
            selected_file = int(input("Please enter the index of the store you want to check (0 for Amazon, 1 for Farmmarket, 2 for Wholefood, 3 for Bestbuy, 4 for Kmart): "))
            if selected_file < 0 or selected_file >= len(datasetlist):
                print("Invalid selection. Please select a valid store.")
                continue
            print(f"You selected store: {datasetlist[selected_file]}")
            confirmation = input("Is this correct? (yes/no): ").strip().lower()
            if confirmation == 'yes':
                min_support = float(input("Enter minimum support (as a decimal between 0 and 1): "))
                if not (0 <= min_support <= 1):
                    raise ValueError("Minimum support must be between 0 and 1.")
                min_confidence = float(input("Enter minimum confidence (as a decimal between 0 and 1): "))
                if not (0 <= min_confidence <= 1):
                    raise ValueError("Minimum confidence must be between 0 and 1.")
                return selected_file, min_support, min_confidence
            elif confirmation == 'no':
                print("Returning to store selection...\n")
                continue
            else:
                print("Invalid input. Please type 'yes' or 'no'.")
        except ValueError as e:
            print(f"Input Error: {e}")
            continue

# Call the function to get user input for store selection, min_support, and min_confidence
selected_file, min_support, min_confidence = select_store_and_input_parameters()

print(f"Proceeding with store: {datasetlist[selected_file]}")
print(f"Minimum support: {min_support}, Minimum confidence: {min_confidence}")

# Open and read the corresponding CSV file
file_name = 'data_' + datasetlist[selected_file] + '.csv'
list_name = 'datalist_' + datasetlist[selected_file] + '.csv'
print(file_name)

# Load the data into a DataFrame
df = pd.read_csv(file_name, encoding='ISO-8859-1')
df_list = pd.read_csv(list_name, encoding='ISO-8859-1')

# Prepare the order list and dataset
order = sorted(df_list['Item.name'].astype(str))
dataset = []

for lines in df['Transaction']:
    trans = [str(item.strip().replace('\x92', "'")) for item in lines.strip().split(',')]
    trans_1 = sorted(np.unique(trans), key=lambda x: order.index(x) if x in order else float('inf'))
    dataset.append(trans_1)

# Brute Force: Function to get frequent itemsets using brute force
def get_frequent_itemsets(dataset, min_support):
    itemset_counts = defaultdict(int)
    num_transactions = len(dataset)

    for transaction in dataset:
        for k in range(1, len(transaction) + 1):  
            for itemset in itertools.combinations(transaction, k):
                itemset_counts[itemset] += 1

    frequent_itemsets = {itemset: count for itemset, count in itemset_counts.items() if count / num_transactions >= min_support}
    return frequent_itemsets

# Function to generate association rules from frequent itemsets
def generate_association_rules(frequent_itemsets, dataset, min_confidence):
    rules = []
    num_transactions = len(dataset)

    for itemset, count in frequent_itemsets.items():
        for k in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, k):
                antecedent = set(antecedent)
                consequent = set(itemset) - antecedent

                if len(consequent) > 0:
                    antecedent_count = sum(1 for transaction in dataset if antecedent.issubset(set(transaction)))
                    rule_support = count
                    rule_confidence = rule_support / antecedent_count if antecedent_count > 0 else 0

                    if rule_confidence >= min_confidence:
                        rules.append((antecedent, consequent, rule_support, rule_confidence))

    return rules

# Measure performance of Brute Force
start_time = time.time()
frequent_itemsets = get_frequent_itemsets(dataset, min_support)
brute_force_rules = generate_association_rules(frequent_itemsets, dataset, min_confidence)


print("\nBrute Force Frequent Itemsets:")
for itemset, count in frequent_itemsets.items():
    print(f"Itemset: {set(map(str, itemset))}, Count: {count}, Support: {count / len(dataset):.4f}")

print("\nBrute Force Association Rules (Antecedent -> Consequent):")
for antecedent, consequent, support, confidence in brute_force_rules:
    antecedent_str = set(map(str, antecedent))
    consequent_str = set(map(str, consequent))
    print(f"{antecedent_str} -> {consequent_str}, Support: {support / len(dataset):.4f}, Confidence: {confidence:.4f}")
brute_force_time = time.time() - start_time
# Measure performance of Apriori algorithm
start_time = time.time()
frequent_itemsets_apriori, apriori_rules = apriori(dataset, minSup=min_support, minConf=min_confidence)

print("\nApriori Frequent Itemsets:")
for support, itemsets in frequent_itemsets_apriori.items():
    for itemset in itemsets:
        itemset_str = set(map(str, itemset))
        print(f"Itemset: {itemset_str}, Support: {support}")

if isinstance(apriori_rules, list) and len(apriori_rules) > 0:
    print("\nApriori Association Rules using apriori-python (Antecedent -> Consequent):")
    for i, rule in enumerate(apriori_rules):
        antecedent = set(map(str, rule[0]))
        consequent = set(map(str, rule[1]))
        confidence = rule[2]
        print(f"Rule {i + 1}: {antecedent} -> {consequent}, Confidence: {confidence:.4f}")
apriori_time = time.time() - start_time
# Measure performance of FP-Growth
start_time = time.time()
def run_fp_growth(dataset, min_support):
    min_support_count = int(min_support * len(dataset))
    patterns = pyfpgrowth.find_frequent_patterns(dataset, min_support_count)
    rules = pyfpgrowth.generate_association_rules(patterns, min_confidence)
    return patterns, rules


patterns_fp_growth, rules_fp_growth = run_fp_growth(dataset, min_support)

print("\nFP-Growth Frequent Itemsets:")
for pattern, count in patterns_fp_growth.items():
    pattern_str = set(map(str, pattern))
    print(f"Itemset: {pattern_str}, Count: {count}")

print("\nFP-Growth Association Rules (Antecedent -> Consequent):")
for antecedent, (consequent, confidence) in rules_fp_growth.items():
    antecedent_str = set(map(str, antecedent))
    consequent_str = set(map(str, consequent))
    print(f"{antecedent_str} -> {consequent_str}, Confidence: {confidence:.4f}")
fp_growth_time = time.time() - start_time

# Performance Summary
print("\nPerformance Summary:")
print(f"Brute Force Execution Time: {brute_force_time:.4f} seconds")
print(f"Apriori Execution Time: {apriori_time:.4f} seconds")
print(f"FP-Growth Execution Time: {fp_growth_time:.4f} seconds")
