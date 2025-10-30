# YOUR CODE HERE
import math

def is_infected(N, D, coordinates):
    infected = [False] * N
    infected[0] = True  # Person 1 is initially infected
    
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    
    # Use a queue to perform BFS
    queue = [0]
    
    while queue:
        current = queue.pop(0)
        for i in range(N):
            if not infected[i] and distance(coordinates[current], coordinates[i]) <= D:
                infected[i] = True
                queue.append(i)
    
    return infected

# Read input
import sys
input = sys.stdin.read
data = input().split()

N = int(data[0])
D = int(data[1])
coordinates = []

for i in range(N):
    X = int(data[2 + 2 * i])
    Y = int(data[3 + 2 * i])
    coordinates.append((X, Y))

infected = is_infected(N, D, coordinates)

for i in range(N):
    if infected[i]:
        print("Yes")
    else:
        print("No")