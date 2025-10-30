# YOUR CODE HERE
import math

def distance(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

N, D = map(int, input().split())
people = [tuple(map(int, input().split())) for _ in range(N)]

infected = [False] * N
infected[0] = True  # Person 1 is initially infected

changed = True
while changed:
    changed = False
    for i in range(N):
        if infected[i]:
            for j in range(N):
                if not infected[j] and distance(people[i][0], people[i][1], people[j][0], people[j][1]) <= D:
                    infected[j] = True
                    changed = True

for is_infected in infected:
    print("Yes" if is_infected else "No")