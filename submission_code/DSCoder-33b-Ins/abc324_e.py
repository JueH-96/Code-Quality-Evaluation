# YOUR CODE HERE

import sys

def is_subsequence(s, t):
    t = iter(t)
    return all(c in t for c in s)

n, t = sys.stdin.readline().split()
n = int(n)
s = [sys.stdin.readline().strip() for _ in range(n)]

count = 0
for i in range(n):
    for j in range(i+1, n):
        if is_subsequence(t, s[i]+s[j]) or is_subsequence(t, s[j]+s[i]):
            count += 1

print(count)