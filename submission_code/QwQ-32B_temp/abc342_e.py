import sys
import heapq

def main():
    sys.setrecursionlimit(1 << 25)
    N, M = map(int, sys.stdin.readline().split())
    edges_by_B = [[] for _ in range(N + 1)]
    edges_by_A = [[] for _ in range(N + 1)]
    for _ in range(M):
        l, d, k, c, A, B = map(int, sys.stdin.readline().split())
        edges_by_B[B].append((l, d, k, c, A))
        edges_by_A[A].append((l, d, k, c, B))

    INF = float('inf')
    f = [-INF] * (N + 1)

    # Process edges leading to N first
    for edge in edges_by_B[N]:
        l, d, k, c, A = edge
        max_t = l + (k - 1) * d
        if max_t > f[A]:
            f[A] = max_t

    # Initialize priority queue with nodes that have f[A] > -INF
    heap = []
    for node in range(1, N + 1):
        if f[node] != -INF:
            heapq.heappush(heap, (-f[node], node))

    processed = [False] * (N + 1)

    while heap:
        current_f_neg, v = heapq.heappop(heap)
        current_f = -current_f_neg
        if processed[v]:
            continue
        processed[v] = True

        # Look at all edges coming into v (edges where B_i = v)
        for edge in edges_by_B[v]:
            l, d, k, c, A = edge
            required = current_f - c
            if required < l:
                continue

            max_departure = l + (k - 1) * d
            if max_departure <= required:
                t_candidate = max_departure
            else:
                numerator = required - l
                if numerator < 0:
                    continue
                m_max = numerator // d
                if numerator % d != 0:
                    m_max = (numerator // d)
                else:
                    m_max = (numerator // d)
                m_max = min(m_max, k - 1)
                if m_max < 0:
                    continue
                t_candidate = l + m_max * d

            if t_candidate < l:
                continue

            if t_candidate > f[A]:
                f[A] = t_candidate
                heapq.heappush(heap, (-f[A], A))

    for S in range(1, N):
        if f[S] == -INF:
            print("Unreachable")
        else:
            print(f[S])

if __name__ == "__main__":
    main()