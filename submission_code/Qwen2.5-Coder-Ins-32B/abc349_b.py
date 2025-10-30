# YOUR CODE HERE
import sys
from collections import Counter

def is_good_string(S):
    count = Counter(S)
    freq_count = Counter(count.values())
    
    for freq in freq_count.values():
        if freq != 0 and freq != 2:
            return "No"
    return "Yes"

S = sys.stdin.read().strip()
print(is_good_string(S))