from collections import Counter

def is_good_string(s):
    letter_counts = Counter(s)
    frequency_counts = Counter(letter_counts.values())
    
    for count in frequency_counts.values():
        if count not in [0, 2]:
            return "No"
    return "Yes"

# Read input from stdin
S = input().strip()

# Check if the string is good and print the result
print(is_good_string(S))