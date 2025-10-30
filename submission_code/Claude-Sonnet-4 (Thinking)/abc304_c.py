from collections import deque

# Read input
n, d = map(int, input().split())
people = []
for i in range(n):
    x, y = map(int, input().split())
    people.append((x, y))

# Function to calculate squared Euclidean distance
def distance_squared(p1, p2):
    return (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2

# BFS to spread infection
infected = [False] * n
infected[0] = True  # Person 1 (index 0) is initially infected
queue = deque([0])
d_squared = d * d

while queue:
    current = queue.popleft()
    current_pos = people[current]
    
    for i in range(n):
        if not infected[i]:
            if distance_squared(current_pos, people[i]) <= d_squared:
                infected[i] = True
                queue.append(i)

# Output results
for i in range(n):
    if infected[i]:
        print("Yes")
    else:
        print("No")