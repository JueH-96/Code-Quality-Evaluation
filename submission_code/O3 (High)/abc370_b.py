import sys

def main() -> None:
    data = list(map(int, sys.stdin.read().strip().split()))
    if not data:
        return
    N = data[0]
    pos = 1

    # store the lower-triangular part (including diagonal)
    A = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(1, i+1):
            A[i][j] = data[pos]
            pos += 1

    cur = 1
    for k in range(1, N+1):
        if cur >= k:
            cur = A[cur][k]
        else:
            cur = A[k][cur]

    print(cur)

if __name__ == "__main__":
    main()