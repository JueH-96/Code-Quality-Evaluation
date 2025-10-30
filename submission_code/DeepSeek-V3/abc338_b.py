# YOUR CODE HERE
from collections import defaultdict

S = input().strip()
freq = defaultdict(int)

for char in S:
    freq[char] += 1

max_freq = max(freq.values())
candidates = [char for char, count in freq.items() if count == max_freq]
candidates.sort()

print(candidates[0])