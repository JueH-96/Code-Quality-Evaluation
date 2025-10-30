def main():
    import sys
    import heapq

    data = sys.stdin.read().strip().split()
    N = int(data[0])

    # We only need A_i, B_i, X_i for i in [1..N-1], so make arrays of length N+1 (index 0 unused)
    A = [0] * (N+1)
    B = [0] * (N+1)
    X = [0] * (N+1)

    idx = 1
    for i in range(1, N):
        A[i] = int(data[idx])
        B[i] = int(data[idx+1])
        X[i] = int(data[idx+2])
        idx += 3

    # Dijkstra-like approach
    INF = 10**20
    dist = [INF] * (N+1)
    dist[1] = 0

    pq = [(0, 1)]  # (cost, stage)

    while pq:
        cost, stage = heapq.heappop(pq)

        if stage == N:
            print(cost)
            return

        if cost > dist[stage]:
            continue

        # If we can go to stage+1 (i -> i+1)
        if stage < N:
            new_cost = cost + A[stage]
            if new_cost < dist[stage+1]:
                dist[stage+1] = new_cost
                heapq.heappush(pq, (new_cost, stage+1))

            # If we can jump to stage X[stage] (i -> X[i])
            jump_cost = cost + B[stage]
            jump_stage = X[stage]
            if jump_cost < dist[jump_stage]:
                dist[jump_stage] = jump_cost
                heapq.heappush(pq, (jump_cost, jump_stage))

    # If somehow stage N wasn't reached during the loop (shouldn't happen in valid inputs),
    # we output dist[N].
    print(dist[N])

# Do not forget to call main
main()