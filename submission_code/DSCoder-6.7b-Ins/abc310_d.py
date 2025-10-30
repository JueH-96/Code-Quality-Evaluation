import sys
from collections import defaultdict
from itertools import combinations

def read_input():
    N, T, M = map(int, sys.stdin.readline().split())
    incompatible = defaultdict(list)
    for _ in range(M):
        a, b = map(int, sys.stdin.readline().split())
        incompatible[a].append(b)
        incompatible[b].append(a)
    return N, T, M, incompatible

def solve():
    N, T, M, incompatible = read_input()
    players = list(range(1, N+1))
    ways = 0
    for team1 in combinations(players, T):
        if T == 1 or len(team1) == N:
            ways += 1
            continue
        for team2 in combinations(list(set(players) - set(team1)), T-1):
            if any(player in team1 for player in team2):
                continue
            ways += 1
    return ways

print(solve())