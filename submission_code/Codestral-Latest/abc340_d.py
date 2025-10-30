import heapq

# Read input
N = int(input())
stages = []
for _ in range(N - 1):
    A, B, X = map(int, input().split())
    stages.append((A, B, X))

# Initialize variables
min_time = [float('inf')] * N
min_time[0] = 0  # Time to reach stage 1 is 0

# Priority queue to explore stages with the minimum time
pq = [(0, 0)]  # (time, stage)

while pq:
    current_time, current_stage = heapq.heappop(pq)

    if current_stage == N - 1:
        break

    # Option 1: Spend A_i seconds to clear stage i and move to stage i+1
    next_stage = current_stage + 1
    time_to_next_stage = current_time + stages[current_stage][0]
    if time_to_next_stage < min_time[next_stage]:
        min_time[next_stage] = time_to_next_stage
        heapq.heappush(pq, (time_to_next_stage, next_stage))

    # Option 2: Spend B_i seconds to clear stage i and move to stage X_i
    next_stage = stages[current_stage][2] - 1
    time_to_next_stage = current_time + stages[current_stage][1]
    if time_to_next_stage < min_time[next_stage]:
        min_time[next_stage] = time_to_next_stage
        heapq.heappush(pq, (time_to_next_stage, next_stage))

# The minimum time to reach stage N
print(min_time[N - 1])