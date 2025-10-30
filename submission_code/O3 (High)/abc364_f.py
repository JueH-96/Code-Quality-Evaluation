import sys
import heapq

def main() -> None:
    sys.setrecursionlimit(1 << 25)
    input = sys.stdin.readline

    N, Q = map(int, input().split())

    # events starting at each position (1-indexed)
    starts = [[] for _ in range(N + 2)]          # starts[x] : list of (cost , right_end)
    diff   = [0] * (N + 3)                       # for checking whether every vertex is covered
    total_cost_extras = 0                        # Σ C_i

    for _ in range(Q):
        L, R, C = map(int, input().split())
        total_cost_extras += C

        # the interval contributes to boundaries j with  L ≤ j ≤ R-1
        starts[L].append((C, R - 1))

        # mark coverage of original vertices
        diff[L] += 1
        diff[R + 1] -= 1

    # make sure every original vertex is incident to at least one edge
    covered = True
    acc = 0
    for i in range(1, N + 1):
        acc += diff[i]
        if acc == 0:
            covered = False
            break
    if not covered:
        print(-1)
        return

    # when N == 1 there are no boundaries, connectivity is already guaranteed
    if N == 1:
        print(total_cost_extras)
        return

    pq = []                         # min-heap of (cost , right_end)
    sum_min_on_boundaries = 0

    for j in range(1, N):           # j corresponds to boundary between j and j+1
        # add intervals starting at position j
        for item in starts[j]:
            heapq.heappush(pq, item)

        # discard intervals that no longer cover this boundary
        while pq and pq[0][1] < j:
            heapq.heappop(pq)

        if not pq:                  # nothing covers boundary → disconnected
            print(-1)
            return

        sum_min_on_boundaries += pq[0][0]

    answer = total_cost_extras + sum_min_on_boundaries
    print(answer)

if __name__ == "__main__":
    main()