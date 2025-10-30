import sys
from collections import Counter

def is_good_string(s):
    # Count the frequency of each character in the string
    char_count = Counter(s)
    # Count the frequency of each frequency
    freq_count = Counter(char_count.values())
    
    # Check if each frequency is either 0 or 2
    for freq in freq_count.values():
        if freq not in [0, 2]:
            return "No"
    return "Yes"

# Read input from stdin
s = input().strip()

# Determine if the string is a good string and print the result
print(is_good_string(s))