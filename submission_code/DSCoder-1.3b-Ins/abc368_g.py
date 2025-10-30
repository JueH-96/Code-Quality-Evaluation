import sys

def solve():
    N = int(sys.stdin.readline().strip())
    A = list(map(int, sys.stdin.readline().strip().split()))
    B = list(map(int, sys.stdin.readline().strip().split()))
    Q = int(sys.stdin.readline().strip())

    v = 0
    for _ in range(Q):
        query = list(map(int, sys.stdin.readline().strip().split()))
        if query[0] == 1:
            v += A[query[1]-1] * query[2]
        elif query[0] == 2:
            v += B[query[1]-1] * query[2]
        else:
            v = max(v + A[query[1]-1], v * B[query[1]-1])

    print(v)

if __name__ == "__main__":
    solve()