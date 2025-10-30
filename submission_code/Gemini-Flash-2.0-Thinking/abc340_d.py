import heapq

def solve():
    n = int(input())
    a = []
    b = []
    x = []
    for _ in range(n - 1):
        ai, bi, xi = map(int, input().split())
        a.append(ai)
        b.append(bi)
        x.append(xi)

    distances = [float('inf')] * (n + 1)
    distances[1] = 0
    pq = [(0, 1)]

    while pq:
        current_distance, current_stage = heapq.heappop(pq)

        if current_distance > distances[current_stage]:
            continue

        if current_stage == n:
            print(current_distance)
            return

        if current_stage < n:
            # Option 1: Go to stage current_stage + 1
            next_stage_1 = current_stage + 1
            time_to_next_stage_1 = a[current_stage - 1]
            if distances[current_stage] + time_to_next_stage_1 < distances[next_stage_1]:
                distances[next_stage_1] = distances[current_stage] + time_to_next_stage_1
                heapq.heappush(pq, (distances[next_stage_1], next_stage_1))

            # Option 2: Go to stage x[current_stage - 1]
            next_stage_2 = x[current_stage - 1]
            time_to_next_stage_2 = b[current_stage - 1]
            if distances[current_stage] + time_to_next_stage_2 < distances[next_stage_2]:
                distances[next_stage_2] = distances[current_stage] + time_to_next_stage_2
                heapq.heappush(pq, (distances[next_stage_2], next_stage_2))

solve()