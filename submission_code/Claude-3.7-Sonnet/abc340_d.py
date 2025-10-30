import heapq

def min_time_to_stage_n(n, actions):
    # Initialize distances to infinity for all stages
    distances = [float('inf')] * (n + 1)
    distances[1] = 0  # Start at stage 1
    
    # Priority queue for Dijkstra's algorithm
    # Each entry is (time_so_far, stage_number)
    pq = [(0, 1)]
    
    while pq:
        time, stage = heapq.heappop(pq)
        
        # If we've reached stage N, return the time
        if stage == n:
            return time
        
        # If we've already found a better path to this stage, skip
        if time > distances[stage]:
            continue
        
        # Only process stages 1 to N-1
        if stage < n:
            a, b, x = actions[stage-1]
            
            # Option 1: Clear stage and go to next stage
            next_time = time + a
            if next_time < distances[stage+1]:
                distances[stage+1] = next_time
                heapq.heappush(pq, (next_time, stage+1))
            
            # Option 2: Clear stage and go to stage X
            next_time = time + b
            if next_time < distances[x]:
                distances[x] = next_time
                heapq.heappush(pq, (next_time, x))
    
    return distances[n]

# Read input and solve the problem
n = int(input())
actions = []
for _ in range(n-1):
    a, b, x = map(int, input().split())
    actions.append((a, b, x))

print(min_time_to_stage_n(n, actions))