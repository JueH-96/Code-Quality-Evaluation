# YOUR CODE HERE
import sys
input = sys.stdin.readline

MOD = 998244353

def modinv(a, m=MOD):
    b, u, v = m, 1, 0
    while b:
        t = a // b
        a -= t * b
        a, b = b, a
        u -= t * v
        u, v = v, u
    u %= m
    if u < 0:
        u += m
    return u

def solve():
    N = int(input())
    wins = [0] * N
    team_size = [1] * N
    for _ in range(N-1):
        p, q = map(int, input().split())
        p -= 1
        q -= 1
        wins[p] += modinv(team_size[p] + team_size[q]) * team_size[q]
        wins[q] += modinv(team_size[p] + team_size[q]) * team_size[p]
        team_size[p] += team_size[q]
        team_size[q] = 0
    for i in range(N):
        wins[i] = (wins[i] + 1) % MOD
    print(' '.join(map(str, wins)))

solve()