import sys

def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    if not data:
        return
    N, M = data[0], data[1]
    A = data[2:2 + N]
    B = data[2 + N: 2 + N + M]

    # Step 1: earliest positions of each B[i] in A
    earliest = [0] * M
    idx_a = 0
    for i in range(M):
        val = B[i]
        while idx_a < N and A[idx_a] != val:
            idx_a += 1
        if idx_a == N:            # B is not a subsequence of A at all
            print("No")
            return
        earliest[i] = idx_a
        idx_a += 1

    # Step 2: latest positions of each B[i] in A
    latest = [0] * M
    idx_a = N - 1
    for i in range(M - 1, -1, -1):
        val = B[i]
        while idx_a >= 0 and A[idx_a] != val:
            idx_a -= 1
        if idx_a < 0:             # This should not happen if Step-1 succeeded
            print("No")
            return
        latest[i] = idx_a
        idx_a -= 1

    # Step 3: check if at least one element can be chosen at a different position
    for i in range(M):
        if earliest[i] < latest[i]:
            print("Yes")
            return
    print("No")


if __name__ == "__main__":
    main()