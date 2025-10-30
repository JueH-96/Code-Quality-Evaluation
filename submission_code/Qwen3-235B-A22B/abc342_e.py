import sys
import heapq

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    idx = 0
    N = int(data[idx])
    idx += 1
    M = int(data[idx])
    idx += 1
    
    rev_adj = [[] for _ in range(N + 1)]  # 1-based indexing

    for _ in range(M):
        l = int(data[idx])
        d = int(data[idx + 1])
        k = int(data[idx + 2])
        c = int(data[idx + 3])
        A = int(data[idx + 4])
        B = int(data[idx + 5])
        idx += 6
        rev_adj[B].append((l, d, k, c, A))

    INF = 10**30
    best = [-INF] * (N + 1)
    best[N] = INF  # best[N] is set to a very large number

    # Use a max-heap (simulated by negative values)
    heap = [(-best[N], N)]
    heapq.heapify(heap)

    while heap:
        neg_time, u = heapq.heappop(heap)
        current_time = -neg_time

        # Skip if this entry is outdated
        if current_time < best[u]:
            continue

        # Process all edges ending at u (i.e., rev_adj[u])
        for (l, d, k, c, A) in rev_adj[u]:
            D = current_time - c
            if D < l:
                continue  # No possible departure time

            # Calculate x_max
            numerator = D - l
            x_max = numerator // d
            if x_max < 0:
                continue
            if x_max >= k:
                x_max = k - 1

            s_candidate = l + x_max * d
            if s_candidate < l:
                continue  # Shouldn't happen due to checks above

            if best[A] < s_candidate:
                best[A] = s_candidate
                heapq.heappush(heap, (-best[A], A))

    # Output the results
    for i in range(1, N):
        if best[i] == -INF:
            print("Unreachable")
        else:
            print(best[i])

if __name__ == "__main__":
    main()