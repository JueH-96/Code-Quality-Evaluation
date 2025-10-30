import sys
from collections import defaultdict

def read_ints(): return map(int, sys.stdin.readline().strip().split())
def read_int(): return int(sys.stdin.readline().strip())

def dfs(person, generation, covered, insurances, tree):
    if generation > insurances[person]:
        return 0
    covered.add(person)
    total_covered = 1
    for child in tree[person]:
        total_covered += dfs(child, generation + 1, covered, insurances, tree)
    return total_covered

N, M = read_ints()
parents = [0] + list(read_ints())
insurances = defaultdict(int)

for _ in range(M):
    x, y = read_ints()
    insurances[x] = max(insurances[x], y)

tree = defaultdict(list)
for i in range(2, N + 1):
    tree[parents[i]].append(i)

covered = set()
dfs(1, 1, covered, insurances, tree)
print(len(covered))