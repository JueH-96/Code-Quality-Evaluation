import sys
import math

# Read input
input = sys.stdin.read
data = input().split()

# Parse input
N = int(data[0])
D = int(data[1])
people = []
for i in range(N):
    x = int(data[2 + 2 * i])
    y = int(data[3 + 2 * i])
    people.append((x, y))

# Initialize infection status
infected = [False] * N
infected[0] = True

# Spread the infection
while True:
    new_infected = False
    for i in range(N):
        if infected[i]:
            for j in range(N):
                if not infected[j]:
                    dx = people[i][0] - people[j][0]
                    dy = people[i][1] - people[j][1]
                    distance = math.sqrt(dx * dx + dy * dy)
                    if distance <= D:
                        infected[j] = True
                        new_infected = True
    if not new_infected:
        break

# Output the results
for i in range(N):
    if infected[i]:
        print("Yes")
    else:
        print("No")