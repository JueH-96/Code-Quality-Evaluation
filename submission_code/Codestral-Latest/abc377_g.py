import sys
from collections import defaultdict

input = sys.stdin.read
data = input().split()

N = int(data[0])
strings = data[1:N+1]

def min_cost_to_match_or_empty(T, previous_strings):
    min_cost = float('inf')
    for prev_str in previous_strings:
        cost = abs(len(T) - len(prev_str))
        for a, b in zip(T, prev_str):
            if a != b:
                cost += 2
        min_cost = min(min_cost, cost)
    return min_cost

previous_strings = set()
results = []

for i in range(N):
    T = strings[i]
    if T in previous_strings:
        results.append(0)
    else:
        cost = min_cost_to_match_or_empty(T, previous_strings)
        results.append(cost)
        previous_strings.add(T)

for result in results:
    print(result)