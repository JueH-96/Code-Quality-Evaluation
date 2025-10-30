def solve():
    import sys
    N_and_S = sys.stdin.read().splitlines()
    N = int(N_and_S[0])
    S = N_and_S[1:N+1]
    M = max(len(s) for s in S)
    T = [['*' for _ in range(N)] for _ in range(M)]
    for i in range(1, N+1):
        s = S[i-1]
        pos = N - i
        for j in range(len(s)):
            if j < M:
                T[j][pos] = s[j]
    for t in T:
        s = ''.join(t).rstrip('*')
        print(s)