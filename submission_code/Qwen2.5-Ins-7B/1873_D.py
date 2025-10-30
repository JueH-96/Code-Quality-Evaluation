# YOUR CODE HERE
from collections import deque

def min_operations(n, k, s):
    operations = 0
    dq = deque()
    for i in range(n):
        if s[i] == 'B':
            dq.append(i)
        if len(dq) == k:
            operations += 1
            dq.clear()
    return operations + bool(dq)

t = int(input())
for _ in range(t):
    n, k = map(int, input().split())
    s = input().strip()
    print(min_operations(n, k, s))