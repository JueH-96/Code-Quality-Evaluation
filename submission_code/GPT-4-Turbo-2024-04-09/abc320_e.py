import sys
import heapq

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])

events = []
index = 2
for _ in range(M):
    T = int(data[index])
    W = int(data[index + 1])
    S = int(data[index + 2])
    events.append((T, W, S))
    index += 3

# This will hold the total noodles each person has got
noodles_received = [0] * N

# Min-heap to manage the availability of people
# Each entry is (available_time, person_index)
available_people = []

# Initialize all people as available at time 0
for i in range(N):
    heapq.heappush(available_people, (0, i))

# Process each event
for T, W, S in events:
    # Remove and collect all people who are available by time T
    while available_people and available_people[0][0] <= T:
        available_time, person_index = heapq.heappop(available_people)
        if available_time <= T:
            # This person gets the noodles and steps out
            noodles_received[person_index] += W
            # They will return at time T + S
            heapq.heappush(available_people, (T + S, person_index))
            break
    else:
        # If no one was available to take the noodles, continue to next event
        continue

# Output the total amount of noodles each person has got
for noodles in noodles_received:
    print(noodles)