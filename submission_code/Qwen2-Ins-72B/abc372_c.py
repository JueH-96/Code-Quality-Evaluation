import sys

N, Q = map(int, input().split())
S = list(input())
T = [0] * N
for i in range(N - 2):
    if S[i] == 'A' and S[i + 1] == 'B' and S[i + 2] == 'C':
        T[i + 2] = 1
for _ in range(Q):
    X, C = input().split()
    X = int(X) - 1
    if S[X] == 'A' and C != 'A':
        if X + 1 < N and S[X + 1] == 'B':
            if X + 2 < N and S[X + 2] == 'C':
                T[X + 2] -= 1
        if X - 1 >= 0 and S[X - 1] == 'B':
            if X - 2 >= 0 and S[X - 2] == 'C':
                T[X - 2] -= 1
    if S[X] == 'B' and C != 'B':
        if X - 1 >= 0 and S[X - 1] == 'A':
            if X + 1 < N and S[X + 1] == 'C':
                T[X + 1] -= 1
        if X + 1 < N and S[X + 1] == 'C':
            if X + 2 < N and C == 'A':
                T[X + 2] += 1
    if S[X] == 'C' and C != 'C':
        if X - 2 >= 0 and S[X - 2] == 'A' and S[X - 1] == 'B':
            T[X - 2] -= 1
        if X - 1 >= 0 and S[X - 1] == 'B':
            if C == 'A':
                T[X] += 1
    if C == 'A':
        if X + 1 < N and S[X + 1] == 'B':
            if X + 2 < N and S[X + 2] == 'C':
                T[X + 2] += 1
    if C == 'B':
        if X - 1 >= 0 and S[X - 1] == 'A':
            if X + 1 < N and S[X + 1] == 'C':
                T[X + 1] += 1
    if C == 'C':
        if X - 2 >= 0 and S[X - 2] == 'A' and S[X - 1] == 'B':
            T[X - 2] += 1
    S[X] = C
    ans = 0
    for i in range(X, N):
        ans += T[i]
        T[i] = ans
    print(ans)