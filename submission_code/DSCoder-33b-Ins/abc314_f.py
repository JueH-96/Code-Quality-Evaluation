# YOUR CODE HERE
import sys

MOD = 998244353

def solve():
    read = sys.stdin.readline
    write = sys.stdout.write
    n = int(read())
    teams = [[i] for i in range(1, n+1)]
    wins = [0]*n
    for _ in range(n-1):
        p, q = map(int, read().split())
        p -= 1
        q -= 1
        p_team = teams[p]
        q_team = teams[q]
        p_len = len(p_team)
        q_len = len(q_team)
        p_prob = p_len / (p_len + q_len)
        q_prob = q_len / (p_len + q_len)
        for player in p_team:
            wins[player] += p_prob
        for player in q_team:
            wins[player] += q_prob
        teams[p] = p_team + q_team
        teams[q] = []
    for i in range(n):
        write(f"{int(wins[i] * MOD + 0.5)}
")

solve()