import heapq

# Read the number of people and the number of occurrences
N, M = map(int, input().split())

# Initialize a priority queue for tracking the return times of people
return_queue = []

# Initialize a list to keep track of the noodles each person has got
noodles_got = [0] * N

# Process each occurrence
for _ in range(M):
    T, W, S = map(int, input().split())

    # Remove people from the queue who have returned before time T
    while return_queue and return_queue[0][0] <= T:
        _, person = heapq.heappop(return_queue)
        heapq.heappush(return_queue, (T, person))

    # If the queue is empty, no one gets the noodles
    if not return_queue:
        heapq.heappush(return_queue, (T + S, 0))
        continue

    # The person at the front of the queue gets the noodles
    _, person = heapq.heappop(return_queue)
    noodles_got[person] += W

    # The person returns to their original position in the queue at time T + S
    heapq.heappush(return_queue, (T + S, person))

# Print the total amount of noodles each person has got
for amount in noodles_got:
    print(amount)