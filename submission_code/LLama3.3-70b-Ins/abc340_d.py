import sys
import heapq

def solve():
    N = int(input())
    A = []
    B = []
    X = []
    for _ in range(N-1):
        a, b, x = map(int, input().split())
        A.append(a)
        B.append(b)
        X.append(x)

    # Create a priority queue to store the stages to be processed
    pq = [(0, 1)]  # (time, stage)

    # Create a dictionary to store the minimum time to reach each stage
    min_time = {1: 0}

    while pq:
        time, stage = heapq.heappop(pq)

        # If we have already found a shorter path to this stage, skip it
        if time > min_time.get(stage, float('inf')):
            continue

        # If we have reached the last stage, print the time and exit
        if stage == N:
            print(time)
            return

        # Try to clear the current stage
        next_stage = stage + 1
        next_time = time + A[stage-1]
        if next_stage not in min_time or next_time < min_time[next_stage]:
            min_time[next_stage] = next_time
            heapq.heappush(pq, (next_time, next_stage))

        # Try to clear the current stage and jump to X[stage-1]
        next_stage = X[stage-1]
        next_time = time + B[stage-1]
        if next_stage not in min_time or next_time < min_time[next_stage]:
            min_time[next_stage] = next_time
            heapq.heappush(pq, (next_time, next_stage))

if __name__ == '__main__':
    solve()