import sys
import heapq

def min_time_to_reach_stage_n():
    input = sys.stdin.read
    data = input().split()

    index = 0
    N = int(data[index])
    index += 1

    A = []
    B = []
    X = []

    for i in range(N-1):
        A.append(int(data[index]))
        index += 1
        B.append(int(data[index]))
        index += 1
        X.append(int(data[index]))
        index += 1

    # Initialize a min-heap priority queue
    pq = [(0, 1)]  # (time, stage)
    visited = [False] * (N + 1)
    min_time = [float('inf')] * (N + 1)
    min_time[1] = 0

    while pq:
        current_time, current_stage = heapq.heappop(pq)

        if current_stage == N:
            print(current_time)
            break

        if visited[current_stage]:
            continue

        visited[current_stage] = True

        if current_stage < N:
            next_stage = current_stage + 1
            new_time = current_time + A[current_stage - 1]
            if new_time < min_time[next_stage]:
                min_time[next_stage] = new_time
                heapq.heappush(pq, (new_time, next_stage))

            teleport_stage = X[current_stage - 1]
            teleport_time = current_time + B[current_stage - 1]
            if teleport_time < min_time[teleport_stage]:
                min_time[teleport_stage] = teleport_time
                heapq.heappush(pq, (teleport_time, teleport_stage))

min_time_to_reach_stage_n()