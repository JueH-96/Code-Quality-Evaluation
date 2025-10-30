# YOUR CODE HERE
import sys
import math
from collections import deque

def main():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    D = int(data[1])
    
    coordinates = []
    index = 2
    for _ in range(N):
        x = int(data[index])
        y = int(data[index + 1])
        coordinates.append((x, y))
        index += 2
    
    # Function to calculate squared distance
    def squared_distance(p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
    
    # D squared
    D_squared = D * D
    
    # BFS to find all infected people
    infected = [False] * N
    infected[0] = True  # Person 1 is initially infected
    
    queue = deque([0])
    
    while queue:
        current = queue.popleft()
        
        for i in range(N):
            if not infected[i]:
                if squared_distance(coordinates[current], coordinates[i]) <= D_squared:
                    infected[i] = True
                    queue.append(i)
    
    # Output results
    for i in range(N):
        if infected[i]:
            print("Yes")
        else:
            print("No")