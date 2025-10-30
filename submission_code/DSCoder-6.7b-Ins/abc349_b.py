# YOUR CODE HERE
import sys
from collections import Counter

def is_good_string(s):
    letter_counts = Counter(s)
    counts = Counter(letter_counts.values())
    return all(count <= 2 for count in counts.values())

s = sys.stdin.readline().strip()
print('Yes' if is_good_string(s) else 'No')