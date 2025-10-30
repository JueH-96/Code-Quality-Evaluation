# YOUR CODE HERE
from collections import Counter

def is_good_string(s):
    letter_counts = Counter(s)
    freq_counts = Counter(letter_counts.values())
    
    for freq, count in freq_counts.items():
        if count not in [0, 2]:
            return "No"
    return "Yes"

s = input().strip()
print(is_good_string(s))