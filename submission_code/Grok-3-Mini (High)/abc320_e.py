import heapq
import sys

# Read all input from stdin
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
M = int(data[index])
index += 1

# Read all drop events
drop_events = []
for _ in range(M):
    T = int(data[index])
    W = int(data[index + 1])
    S = int(data[index + 2])
    index += 3
    drop_events.append((T, W, S))

# Initialize event heap with all drop events
event_heap = [(drop_events[i][0], 1, i) for i in range(M)]  # (time, action_code, index)
heapq.heapify(event_heap)

# Initialize present heap with all people present
present_heap = [i for i in range(1, N + 1)]  # Person IDs from 1 to N
heapq.heapify(present_heap)

# Initialize answer array
ans = [0] * (N + 1)  # Index 0 unused, ans[1] to ans[N] for persons

# Process all events
while event_heap:
    time, action_code, extra = heapq.heappop(event_heap)
    if action_code == 1:  # Drop event
        i = extra  # Index of drop event
        W = drop_events[i][1]
        S = drop_events[i][2]
        if present_heap:  # If someone is present
            person = heapq.heappop(present_heap)  # Get and remove the min (front)
            ans[person] += W  # Add noodles to person
            # Schedule return event
            return_time = time + S
            heapq.heappush(event_heap, (return_time, 0, person))  # (time, action_code=0 for return, person_id)
    elif action_code == 0:  # Return event
        person = extra  # Person ID
        heapq.heappush(present_heap, person)  # Add person back to present heap

# Output the answer for each person
for i in range(1, N + 1):
    print(ans[i])