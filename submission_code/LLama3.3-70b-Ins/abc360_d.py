import sys

def solve():
    N, T = map(int, sys.stdin.readline().split())
    S = sys.stdin.readline().strip()
    X = list(map(int, sys.stdin.readline().split()))

    ants = []
    for i in range(N):
        ants.append((X[i], int(S[i]) * 2 - 1))

    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            if (ants[i][0] - ants[j][0]) * (ants[i][1] - ants[j][1]) < 0:
                if abs(ants[i][0] - ants[j][0]) <= T + 0.1:
                    count += 1

    print(count)

if __name__ == "__main__":
    solve()