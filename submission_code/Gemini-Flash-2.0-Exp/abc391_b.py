def solve():
    N, M = map(int, input().split())
    S = []
    for _ in range(N):
        S.append(input())
    T = []
    for _ in range(M):
        T.append(input())

    for a in range(1, N - M + 2):
        for b in range(1, N - M + 2):
            match = True
            for i in range(1, M + 1):
                for j in range(1, M + 1):
                    if S[a+i-2][b+j-2] != T[i-1][j-1]:
                        match = False
                        break
                if not match:
                    break
            if match:
                print(a, b)
                return

solve()