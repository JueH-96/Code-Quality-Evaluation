import sys
import heapq

# ------------------------------------------------------------
def main() -> None:
    data = list(map(int, sys.stdin.buffer.read().split()))
    N, K = data[0], data[1]
    A = data[2          : 2 + N]
    B = data[2 + N      : 2 + 2 * N]
    C = data[2 + 2 * N  : 2 + 3 * N]

    A.sort(reverse=True)
    B.sort(reverse=True)
    C.sort(reverse=True)

    def value(i: int, j: int, k: int) -> int:
        return A[i] * B[j] + B[j] * C[k] + C[k] * A[i]

    # priority queue (max-heap implemented with negatives)
    h: list[tuple[int, int, int, int]] = []
    first_val = value(0, 0, 0)
    heapq.heappush(h, (-first_val, 0, 0, 0))

    visited: set[int] = set()
    visited.add(0)                      # code(0,0,0) == 0

    def code(i: int, j: int, k: int) -> int:
        return (i * N + j) * N + k

    ans = 0
    for _ in range(K):
        neg, i, j, k = heapq.heappop(h)
        ans = -neg

        # push the three neighbours
        if i + 1 < N:
            c = code(i + 1, j, k)
            if c not in visited:
                visited.add(c)
                heapq.heappush(h, (-value(i + 1, j, k), i + 1, j, k))

        if j + 1 < N:
            c = code(i, j + 1, k)
            if c not in visited:
                visited.add(c)
                heapq.heappush(h, (-value(i, j + 1, k), i, j + 1, k))

        if k + 1 < N:
            c = code(i, j, k + 1)
            if c not in visited:
                visited.add(c)
                heapq.heappush(h, (-value(i, j, k + 1), i, j, k + 1))

    print(ans)

# ------------------------------------------------------------
if __name__ == "__main__":
    main()