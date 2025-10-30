import math
from itertools import product

n, t, m = map(int, input().split())
forbidden = set()
for _ in range(m):
    a, b = map(int, input().split())
    forbidden.add(frozenset({a-1, b-1}))

count = 0
for assignment in product(range(t), repeat=n):
    # Check all teams are present
    if len(set(assignment)) != t:
        continue
    # Check forbidden pairs
    valid = True
    for pair in forbidden:
        a, b = pair
        if assignment[a] == assignment[b]:
            valid = False
            break
    if valid:
        count += 1

answer = count // math.factorial(t)
print(answer)