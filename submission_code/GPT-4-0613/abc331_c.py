import sys
from collections import Counter

N = int(sys.stdin.readline())
A = list(map(int, sys.stdin.readline().split()))

counter = Counter(A)
keys = sorted(counter.keys(), reverse=True)
cumulative = [0] * (len(keys) + 1)

for i in range(len(keys) - 1, -1, -1):
    cumulative[i] = cumulative[i + 1] + keys[i] * counter[keys[i]]

for a in A:
    print(cumulative[keys.index(a) + 1])