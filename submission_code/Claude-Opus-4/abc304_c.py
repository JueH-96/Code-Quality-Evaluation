# YOUR CODE HERE
import math

# Read input
N, D = map(int, input().split())
people = []
for i in range(N):
    x, y = map(int, input().split())
    people.append((x, y))

# Track infected people
infected = [False] * N
infected[0] = True  # Person 1 (index 0) is initially infected

# Keep spreading until no new infections
changed = True
while changed:
    changed = False
    for i in range(N):
        if infected[i]:
            # Check all other people
            for j in range(N):
                if not infected[j]:
                    # Calculate distance
                    x1, y1 = people[i]
                    x2, y2 = people[j]
                    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                    
                    # If within distance D, infect them
                    if dist <= D:
                        infected[j] = True
                        changed = True

# Output results
for i in range(N):
    if infected[i]:
        print("Yes")
    else:
        print("No")