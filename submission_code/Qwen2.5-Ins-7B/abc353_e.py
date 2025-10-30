# N = int(input())
# S = [input() for _ in range(N)]

N = 3
S = ["ab", "abc", "arc"]

from collections import defaultdict

def f(x, y):
    i = 0
    while i < min(len(x), len(y)) and x[i] == y[i]:
        i += 1
    return i

prefix_sum = defaultdict(int)
for s in S:
    for i in range(len(s)):
        prefix_sum[s[:i+1]] += 1

total = 0
for s in S:
    for i in range(len(s)):
        total += prefix_sum[s[:i+1]] - 1

print(total)