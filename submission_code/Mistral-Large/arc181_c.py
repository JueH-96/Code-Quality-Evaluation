import sys

def solve():
    input = sys.stdin.read
    data = input().split()

    N = int(data[0])
    P = list(map(int, data[1:N+1]))
    Q = list(map(int, data[N+1:2*N+1]))

    grid = [['' for _ in range(N)] for _ in range(N)]

    order = [(i, j) for i in range(N) for j in range(N)]
    order.sort(key=lambda x: (P[x[0]], Q[x[1]]))

    value = 0
    for i, j in order:
        grid[i][j] = str(value)
        value = 1 - value

    for row in grid:
        print(''.join(row))

if __name__ == "__main__":
    solve()