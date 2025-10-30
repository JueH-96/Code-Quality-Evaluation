# YOUR CODE HERE

import sys
from collections import Counter

S = sys.stdin.readline().strip()

counter = Counter(S)
max_count = max(counter.values())

most_frequent_chars = [char for char, count in counter.items() if count == max_count]
most_frequent_chars.sort()

print(most_frequent_chars[0])