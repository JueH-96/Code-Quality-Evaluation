import heapq
import bisect

N, M = map(int, input().split())
noodles = [0] * (N + 1)  # noodles[i] is the amount person i has got
row = list(range(1, N + 1))  # initially, everyone is in the row
returns = []  # heap of (return_time, person) for people who are out

for _ in range(M):
    T, W, S = map(int, input().split())
    
    # Process returns - anyone returning at or before time T
    while returns and returns[0][0] <= T:
        return_time, person = heapq.heappop(returns)
        # Insert person back maintaining original order
        bisect.insort(row, person)
    
    # Give noodles to the front person
    if row:
        front_person = row.pop(0)
        noodles[front_person] += W
        # Schedule their return
        heapq.heappush(returns, (T + S, front_person))

# Print results
for i in range(1, N + 1):
    print(noodles[i])