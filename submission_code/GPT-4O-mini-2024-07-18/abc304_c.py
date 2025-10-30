import sys
import math
from collections import deque

def solve():
    input = sys.stdin.read
    data = input().splitlines()
    
    N, D = map(int, data[0].split())
    points = [tuple(map(int, line.split())) for line in data[1:N+1]]
    
    D_squared = D * D  # We will compare squared distances to avoid sqrt for efficiency
    infected = [False] * N
    infected[0] = True  # Person 1 is initially infected
    
    # BFS or DFS to spread the infection
    queue = deque([0])  # Start with person 1 (index 0)
    
    while queue:
        current = queue.popleft()
        
        for i in range(N):
            if not infected[i]:  # If person i is not infected
                # Calculate squared distance
                dist_squared = (points[current][0] - points[i][0]) ** 2 + (points[current][1] - points[i][1]) ** 2
                if dist_squared <= D_squared:  # If within distance D
                    infected[i] = True
                    queue.append(i)  # Add to queue to spread the infection
    
    # Output results
    for i in range(N):
        print("Yes" if infected[i] else "No")