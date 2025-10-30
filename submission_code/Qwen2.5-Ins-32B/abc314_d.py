import sys

def solve():
    N = int(input())
    S = list(input())
    Q = int(input())
    for _ in range(Q):
        t, x, c = input().split()
        t, x = int(t), int(x)
        if t == 1:
            S[x-1] = c
        elif t == 2:
            S = [char.lower() for char in S]
        elif t == 3:
            S = [char.upper() for char in S]
    print(''.join(S))

solve()