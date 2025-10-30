# YOUR CODE HERE
import heapq

N, M = map(int, input().split())

# Track noodles received by each person
noodles = [0] * (N + 1)  # 1-indexed

# Priority queue for people returning (time, person_id)
returns = []

# Current queue of people (those who are in line)
queue = list(range(1, N + 1))

for _ in range(M):
    T, W, S = map(int, input().split())
    
    # Process all returns that happen before or at time T
    while returns and returns[0][0] <= T:
        return_time, person = heapq.heappop(returns)
        queue.append(person)
    
    # If there's someone in the queue
    if queue:
        # Person at front gets the noodles
        person = queue.pop(0)
        noodles[person] += W
        
        # Schedule their return
        heapq.heappush(returns, (T + S, person))

# Output the results
for i in range(1, N + 1):
    print(noodles[i])