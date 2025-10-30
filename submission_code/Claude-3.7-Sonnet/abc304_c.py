import math
from collections import deque

def solve():
    # Read input
    N, D = map(int, input().split())
    coordinates = []
    for _ in range(N):
        x, y = map(int, input().split())
        coordinates.append((x, y))
    
    # BFS to find all infected people
    infected = [False] * N
    queue = deque([0])  # Start with person 1 (0-indexed)
    infected[0] = True
    
    while queue:
        current = queue.popleft()
        for i in range(N):
            if not infected[i]:
                # Calculate Euclidean distance
                x1, y1 = coordinates[current]
                x2, y2 = coordinates[i]
                dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
                
                if dist <= D:
                    infected[i] = True
                    queue.append(i)
    
    # Output results
    for is_infected in infected:
        print("Yes" if is_infected else "No")

solve()