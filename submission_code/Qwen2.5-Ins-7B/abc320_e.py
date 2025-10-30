# YOUR CODE HERE
from collections import defaultdict

N, M = map(int, input().split())
events = []
for _ in range(M):
    T, W, S = map(int, input().split())
    events.append((T, W, S))

events.sort()
people = defaultdict(int)
times = [0] * (N + 1)

for T, W, S in events:
    while times[1] < T:
        for i in range(1, N + 1):
            if times[i] <= times[i - 1]:
                people[i] += times[i - 1] - times[i]
        times[1] += 1
    people[1] += W
    times[1] = T + S

for i in range(1, N + 1):
    print(people[i])