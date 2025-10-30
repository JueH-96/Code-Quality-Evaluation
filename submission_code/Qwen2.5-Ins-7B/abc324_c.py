# YOUR CODE HERE
from collections import deque

def is_valid(s, t):
    if len(s) == len(t):
        return s == t
    if len(s) + 1 != len(t):
        return False
    i, j = 0, 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            i += 1
        j += 1
    return i == len(s)

n, t = input().split()
t = deque(t)
result = []
for i in range(int(n)):
    s = input().split()[0]
    if len(s) == len(t) and s == t or \
       (len(s) + 1 == len(t) and is_valid(s, t)) or \
       (len(s) - 1 == len(t) and is_valid(t, s)) or \
       (len(s) == len(t) + 1 and is_valid(t, s)):
        result.append(i + 1)
print(len(result))
print(*result)