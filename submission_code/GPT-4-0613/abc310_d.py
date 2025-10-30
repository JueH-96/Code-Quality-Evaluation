from itertools import combinations
from collections import defaultdict

def solve():
    N, T, M = map(int, input().split())
    pairs = [tuple(map(int, input().split())) for _ in range(M)]
    incompatible = defaultdict(list)
    for a, b in pairs:
        incompatible[a].append(b)
        incompatible[b].append(a)
    ways = 0
    for teams in combinations(range(1, N+1), T):
        for team in teams:
            if any(other in teams for other in incompatible[team]):
                break
        else:
            ways += 1
    print(ways)

solve()