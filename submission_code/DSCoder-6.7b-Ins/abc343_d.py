import sys
from collections import defaultdict

def solve():
    n, t = map(int, sys.stdin.readline().split())
    players = defaultdict(set)
    scores = defaultdict(int)
    for _ in range(t):
        a, b = map(int, sys.stdin.readline().split())
        players[a].add(b)
    for player in sorted(players.keys()):
        for score in players[player]:
            scores[player] += score
            print(len(scores))

solve()