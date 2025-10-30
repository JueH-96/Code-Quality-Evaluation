from bisect import insort
from collections import Counter

N, M, S_x, S_y = map(int, input().split())
houses = []
for _ in range(N):
    houses.append(tuple(map(int, input().split())))

visited = [(S_x, S_y)]
for _ in range(M):
    D, C = input().split()
    C = int(C)
    if D == "U":
        S_y += C
        insort(visited, (S_x, S_y))
    elif D == "D":
        S_y -= C
        insort(visited, (S_x, S_y))
    elif D == "L":
        S_x -= C
        insort(visited, (S_x, S_y))
    elif D == "R":
        S_x += C
        insort(visited, (S_x, S_y))

# Count unique houses passed or arrived at
count = Counter(visited) + Counter(houses)
total_houses_visited = len([house for house, cnt in count.items() if cnt > 0])

print(S_x, S_y, total_houses_visited)