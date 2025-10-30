import sys
input = sys.stdin.read

def solve():
    data = input().split()
    N = int(data[0])
    T = int(data[1])
    S = data[2]
    X = list(map(int, data[3:]))

    ants = [(X[i], S[i]) for i in range(N)]
    ants.sort()

    count = 0
    for i in range(N):
        for j in range(i + 1, N):
            xi, di = ants[i]
            xj, dj = ants[j]
            if di == '1' and dj == '0':
                t = (xj - xi) / 2.0
                if 0 <= t <= T + 0.1:
                    count += 1

    print(count)

solve()