import sys

def read_input():
    N = int(sys.stdin.readline().strip())
    A = []
    for _ in range(N):
        row = []
        for _ in range(N):
            row.append(list(map(int, sys.stdin.readline().strip().split())))
        A.append(row)
    Q = int(sys.stdin.readline().strip())
    queries = []
    for _ in range(Q):
        queries.append(list(map(int, sys.stdin.readline().strip().split())))
    return N, A, Q, queries

def solve(N, A, Q, queries):
    for query in queries:
        Lx, Rx, Ly, Ry, Lz, Rz = query
        total = 0
        for x in range(Lx-1, Rx):
            for y in range(Ly-1, Ry):
                for z in range(Lz-1, Rz):
                    total += A[x][y][z]
        print(total)

def main():
    N, A, Q, queries = read_input()
    solve(N, A, Q, queries)

if __name__ == "__main__":
    main()