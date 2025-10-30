import sys
from collections import deque

def canReachCorner(xCorner: int, yCorner: int, circles: list[list[int]]) -> bool:
    nodes = [(0, 0), (xCorner, yCorner)]
    for c in circles:
        nodes.append((c[0], c[1]))
    
    graph = {}
    for i in range(len(nodes)):
        graph.setdefault(nodes[i], []).append(nodes[i])
    
    for i in range(len(nodes)):
        A = nodes[i]
        for j in range(i + 1, len(nodes)):
            B = nodes[j]
            blocked = False
            for circle in circles:
                xc, yc, r = circle[0], circle[1], circle[2]
                # Line equation: a*x + b*y + c = 0
                a = B[1] - A[1]
                b = A[0] - B[0]
                c_val = B[0] * A[1] - A[0] * B[1]
                numerator = abs(a * xc + b * yc + c_val)
                denominator = (a**2 + b**2)**0.5
                if denominator == 0:
                    continue
                distance = numerator / denominator
                if distance > r:
                    continue
                # Compute projection of (xc, yc) onto line AB
                dx = B[0] - A[0]
                dy = B[1] - A[1]
                t_numerator = (xc - A[0]) * dx + (yc - A[1]) * dy
                t_denominator = dx**2 + dy**2
                if t_denominator == 0:
                    continue
                t = t_numerator / t_denominator
                t = max(0, min(t, 1))
                blocked = True
                break
            if not blocked:
                if A not in graph:
                    graph[A] = []
                if B not in graph:
                    graph[B] = []
                graph[A].append(B)
                graph[B].append(A)
    
    visited = set()
    queue = deque()
    start = (0, 0)
    if start in graph:
        queue.append(start)
        visited.add(start)
    
    while queue:
        current = queue.popleft()
        if current == (xCorner, yCorner):
            return True
        for neighbor in graph.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    return False