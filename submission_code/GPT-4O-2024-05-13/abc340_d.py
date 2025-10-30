# YOUR CODE HERE
import heapq
import sys
input = sys.stdin.read

def min_time_to_stage_n(N, stages):
    # Initialize the minimum time array with infinity
    min_time = [float('inf')] * (N + 1)
    min_time[1] = 0  # Starting at stage 1 with 0 seconds
    
    # Priority queue to process stages in the order of minimum time
    pq = [(0, 1)]  # (time, stage)
    
    while pq:
        current_time, current_stage = heapq.heappop(pq)
        
        if current_stage == N:
            return current_time
        
        if current_time > min_time[current_stage]:
            continue
        
        # Option 1: Clear current stage to move to the next stage
        if current_stage < N:
            next_stage = current_stage + 1
            next_time = current_time + stages[current_stage - 1][0]
            if next_time < min_time[next_stage]:
                min_time[next_stage] = next_time
                heapq.heappush(pq, (next_time, next_stage))
        
        # Option 2: Clear current stage to move to stage X_i
        next_stage = stages[current_stage - 1][2]
        next_time = current_time + stages[current_stage - 1][1]
        if next_time < min_time[next_stage]:
            min_time[next_stage] = next_time
            heapq.heappush(pq, (next_time, next_stage))
    
    return min_time[N]

def main():
    data = input().split()
    N = int(data[0])
    stages = []
    index = 1
    for i in range(N - 1):
        A = int(data[index])
        B = int(data[index + 1])
        X = int(data[index + 2])
        stages.append((A, B, X))
        index += 3
    
    result = min_time_to_stage_n(N, stages)
    print(result)

if __name__ == "__main__":
    main()