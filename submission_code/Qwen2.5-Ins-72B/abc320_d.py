# YOUR CODE HERE
import sys
from collections import defaultdict, deque

def solve():
    input = sys.stdin.read
    data = input().split()
    
    N = int(data[0])
    M = int(data[1])
    
    graph = defaultdict(list)
    positions = defaultdict(lambda: (None, None))
    indegrees = defaultdict(int)
    
    for i in range(M):
        A = int(data[2 + 4 * i])
        B = int(data[3 + 4 * i])
        X = int(data[4 + 4 * i])
        Y = int(data[5 + 4 * i])
        
        graph[A].append((B, X, Y))
        indegrees[B] += 1
    
    queue = deque()
    positions[1] = (0, 0)
    
    for i in range(1, N + 1):
        if indegrees[i] == 0:
            queue.append(i)
    
    while queue:
        current = queue.popleft()
        current_x, current_y = positions[current]
        
        for neighbor, dx, dy in graph[current]:
            if positions[neighbor] == (None, None):
                positions[neighbor] = (current_x + dx, current_y + dy)
                indegrees[neighbor] -= 1
                if indegrees[neighbor] == 0:
                    queue.append(neighbor)
    
    for i in range(1, N + 1):
        if positions[i] == (None, None):
            print("undecidable")
        else:
            print(*positions[i])

solve()