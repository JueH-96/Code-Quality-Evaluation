import sys

def read_input():
    N = int(sys.stdin.readline().strip())
    P = [int(x) for x in sys.stdin.readline().strip().split()]
    Q = [int(x) for x in sys.stdin.readline().strip().split()]
    return N, P, Q

def solve(N, P, Q):
    grid = [['0' for _ in range(N)] for _ in range(N)]

    # Construct the rows
    for i in range(N):
        row = ''.join(['1' if j <= i else '0' for j in range(N)])
        grid[P[i] - 1] = list(row)

    # Construct the columns
    for i in range(N):
        col = ''.join(['0' if j <= i else '1' for j in range(N)])
        for j in range(N):
            grid[j][Q[i] - 1] = col[j]

    return '
'.join([''.join(row) for row in grid])

if __name__ == '__main__':
    N, P, Q = read_input()
    print(solve(N, P, Q))