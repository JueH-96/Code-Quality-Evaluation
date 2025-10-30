import sys
from itertools import accumulate

def main():
    input = sys.stdin.readline
    N, Q = map(int, input().split())
    P = [list(input().strip()) for _ in range(N)]
    S = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(N):
        for j in range(N):
            S[i + 1][j + 1] = S[i + 1][j] + S[i][j + 1] - S[i][j] + (P[i][j] == 'B')
    for _ in range(Q):
        A, B, C, D = map(int, input().split())
        A, B, C, D = A - 1, B - 1, C - 1, D - 1
        A, B, C, D = A % N, B % N, C % N, D % N
        ans = 0
        for i in range(A, C + 1, N):
            for j in range(B, D + 1, N):
                ans += S[min(i + N, N)][min(j + N, N)] - S[min(i + N, N)][j] - S[i][min(j + N, N)] + S[i][j]
        print(ans)

main()