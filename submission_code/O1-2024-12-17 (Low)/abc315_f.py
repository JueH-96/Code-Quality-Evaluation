import sys
import math
import heapq

def main():
    input_data = sys.stdin.read().strip().split()
    N = int(input_data[0])
    coords = []
    idx = 1
    for _ in range(N):
        x = float(input_data[idx]); y = float(input_data[idx+1])
        coords.append((x,y))
        idx += 2

    # If N=2, then we must visit both, no skips possible, answer is just distance(1,2).
    if N == 2:
        x1,y1 = coords[0]
        x2,y2 = coords[1]
        print(math.hypot(x2 - x1, y2 - y1))
        return

    # Precompute distances for "up to 31 steps" jumps
    # dist_arr[i][d] = distance from checkpoint i to i+d (0-based index), for 1 <= d <= 31
    # if i+d < N
    max_skip = 30  # we'll only allow up to 30 total skips, as beyond that penalty becomes too large to help.
    dist_arr = [[] for _ in range(N)]
    for i in range(N):
        dist_arr[i] = [0.0]* (min(max_skip+1, N - i))
        for d in range(1, len(dist_arr[i])):
            x1, y1 = coords[i]
            x2, y2 = coords[i + d]
            dist_arr[i][d] = math.hypot(x2 - x1, y2 - y1)

    # dp[i][c] = minimal distance (no penalty included) to reach checkpoint i (0-based) having used c skips in total
    # We'll use D' (Dijkstra-like) since the costs are positive real values.
    INF = float('inf')
    dp = [[INF]*(max_skip+1) for _ in range(N)]
    dp[0][0] = 0.0  # start at checkpoint 0 with 0 cost and 0 skips

    # Min-heap of (cost, i, c) meaning we've reached i with c skips at cost
    heap = [(0.0, 0, 0)]
    while heap:
        cur_cost, i, c = heapq.heappop(heap)
        if cur_cost > dp[i][c]:
            continue
        if i == N-1:
            # Already at the last checkpoint, no need to explore further from here
            continue
        # Try jumping forward up to 31 steps (or up to the end)
        # If we jump from i to i+d, we skip (d-1) checkpoints
        for d in range(1, len(dist_arr[i])):  # up to min(31, N - i - 1)
            c_next = c + (d - 1)
            if c_next > max_skip:
                break
            i_next = i + d
            cost_next = cur_cost + dist_arr[i][d]
            if cost_next < dp[i_next][c_next]:
                dp[i_next][c_next] = cost_next
                heapq.heappush(heap, (cost_next, i_next, c_next))

    # Now compute the final answer:
    # penalty(c) = 2^(c-1) if c>0, else 0
    best = dp[N-1][0]  # c=0 has zero penalty
    for c in range(1, max_skip+1):
        if dp[N-1][c] < INF:
            # 2^(c-1)
            penalty = 2.0**(c-1)
            val = dp[N-1][c] + penalty
            if val < best:
                best = val

    # Print result with the required precision
    print(f"{best:.9f}")

# Don't forget to call main()
if __name__ == "__main__":
    main()