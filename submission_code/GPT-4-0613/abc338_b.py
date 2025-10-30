# YOUR CODE HERE
from collections import Counter

s = input().strip()
counter = Counter(s)
max_freq = max(counter.values())
candidates = [k for k, v in counter.items() if v == max_freq]
print(min(candidates))