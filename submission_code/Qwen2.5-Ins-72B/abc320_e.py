# YOUR CODE HERE
import sys
from heapq import heappop, heappush

input = sys.stdin.read
data = input().split()

N = int(data[0])
M = int(data[1])
events = []
for i in range(M):
    T = int(data[2 + 3 * i])
    W = int(data[3 + 3 * i])
    S = int(data[4 + 3 * i])
    events.append((T, W, S))

noodles = [0] * N
queue = list(range(N))
time = 0
return_times = []

for T, W, S in events:
    while return_times and return_times[0][0] <= T:
        _, person = heappop(return_times)
        queue.append(person)
    
    if queue:
        person = queue.pop(0)
        noodles[person] += W
        heappush(return_times, (T + S, person))

for noodle in noodles:
    print(noodle)