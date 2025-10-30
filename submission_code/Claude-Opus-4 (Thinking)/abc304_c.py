from collections import deque

# Read input
N, D = map(int, input().split())
people = []
for i in range(N):
    x, y = map(int, input().split())
    people.append((x, y))

# Helper function to calculate Euclidean distance
def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5

# Track infected people
infected = [False] * N
infected[0] = True  # Person 1 is initially infected

# BFS to spread infection
queue = deque([0])

while queue:
    current = queue.popleft()
    # Check all other people
    for i in range(N):
        if not infected[i] and distance(people[current], people[i]) <= D:
            infected[i] = True
            queue.append(i)

# Output results
for i in range(N):
    if infected[i]:
        print("Yes")
    else:
        print("No")