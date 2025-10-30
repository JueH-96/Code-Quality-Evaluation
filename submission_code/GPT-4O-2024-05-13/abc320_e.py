# YOUR CODE HERE
import heapq
import sys
input = sys.stdin.read

data = input().split()
N = int(data[0])
M = int(data[1])

events = []
for i in range(M):
    T = int(data[2 + 3 * i])
    W = int(data[2 + 3 * i + 1])
    S = int(data[2 + 3 * i + 2])
    events.append((T, W, S))

# Initialize the amount of noodles each person has got
noodles = [0] * N

# Priority queue to manage the people in the row
row = [(0, i) for i in range(N)]
heapq.heapify(row)

# Priority queue to manage the people returning to the row
return_queue = []

for T, W, S in events:
    # Process all people returning to the row at or before time T
    while return_queue and return_queue[0][0] <= T:
        return_time, person = heapq.heappop(return_queue)
        heapq.heappush(row, (return_time, person))
    
    if row:
        _, person = heapq.heappop(row)
        noodles[person] += W
        heapq.heappush(return_queue, (T + S, person))

# Print the result
for amount in noodles:
    print(amount)