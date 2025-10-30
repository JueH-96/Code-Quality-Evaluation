# YOUR CODE HERE
import heapq

N, M = map(int, input().split())

can_opener_counts = [0] * (N + 1)
can_happiness = []
pull_tab_happiness = []

for _ in range(N):
    T, X = map(int, input().split())
    if T == 0:
        heapq.heappush(pull_tab_happiness, -X)
    elif T == 1:
        heapq.heappush(can_happiness, -X)
    else:
        can_opener_counts[X] += 1

for i in range(N - 1, 0, -1):
    can_opener_counts[i] += can_opener_counts[i + 1]

total_happiness = 0
for _ in range(M):
    if can_happiness and can_opener_counts[1] > 0:
        total_happiness -= heapq.heappop(can_happiness)
        can_opener_counts[1] -= 1
    elif pull_tab_happiness:
        total_happiness -= heapq.heappop(pull_tab_happiness)
    elif can_happiness:
        total_happiness -= heapq.heappop(can_happiness)

print(total_happiness)