import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
X1 = int(data[2])

trains = []
index = 3
for i in range(M):
    A = int(data[index])
    B = int(data[index+1])
    S = int(data[index+2])
    T = int(data[index+3])
    trains.append((A, B, S, T, i))
    index += 4

# Sort trains by their start time
trains.sort(key=lambda x: x[2])

# Graph to hold outgoing trains from each city
outgoing = {i: [] for i in range(1, N+1)}
# Graph to hold incoming trains to each city
incoming = {i: [] for i in range(1, N+1)}

for train in trains:
    A, B, S, T, idx = train
    outgoing[A].append((S, T, idx))
    incoming[B].append((S, T, idx))

# Delay times for each train, initialized to None
delays = [None] * M
delays[0] = X1

# Priority queue for processing, starting with the first train
pq = [(X1, 0)]  # (delay, index)

while pq:
    current_delay, idx = heapq.heappop(pq)
    
    # Get the current train details
    _, B, _, T, _ = trains[idx]
    
    # Check all trains that start from city B
    if B in outgoing:
        for S_next, T_next, idx_next in outgoing[B]:
            if T + current_delay <= S_next:
                # Calculate the required delay for the next train
                required_delay = max(0, current_delay + T - S_next)
                if delays[idx_next] is None or required_delay < delays[idx_next]:
                    delays[idx_next] = required_delay
                    heapq.heappush(pq, (required_delay, idx_next))

# Output the delays for train 2 to M
print(" ".join(str(delays[i] if delays[i] is not None else 0) for i in range(1, M)))