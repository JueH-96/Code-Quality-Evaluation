def minimum_time_to_stage_n(N, stages):
    import heapq
    
    # Initialize the minimum time to reach each stage
    min_time = [float('inf')] * (N + 1)
    min_time[1] = 0  # Starting at stage 1 takes 0 seconds
    
    # Priority queue for processing stages based on minimum time
    pq = [(0, 1)]  # (time, stage)
    
    while pq:
        current_time, current_stage = heapq.heappop(pq)
        
        # If we already found a better way to this stage, skip it
        if current_time > min_time[current_stage]:
            continue
        
        # If we are at the last stage, we can stop
        if current_stage == N:
            return current_time
        
        # Get the actions for the current stage
        if current_stage < N:
            A_i, B_i, X_i = stages[current_stage - 1]
            
            # Action 1: Move to stage i + 1
            next_stage = current_stage + 1
            new_time = current_time + A_i
            if new_time < min_time[next_stage]:
                min_time[next_stage] = new_time
                heapq.heappush(pq, (new_time, next_stage))
            
            # Action 2: Move to stage X_i
            next_stage = X_i
            new_time = current_time + B_i
            if new_time < min_time[next_stage]:
                min_time[next_stage] = new_time
                heapq.heappush(pq, (new_time, next_stage))
    
    return min_time[N]

# Read input
import sys
input = sys.stdin.read
data = input().splitlines()

N = int(data[0])
stages = [tuple(map(int, line.split())) for line in data[1:]]

# Calculate and print the result
result = minimum_time_to_stage_n(N, stages)
print(result)