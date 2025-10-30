import math
from collections import deque
from typing import List

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        def is_point_in_circle(x, y, circle):
            cx, cy, r = circle
            return (x - cx) ** 2 + (y - cy) ** 2 <= r * r
        
        if any(is_point_in_circle(0, 0, circle) for circle in circles):
            return False
        if any(is_point_in_circle(xCorner, yCorner, circle) for circle in circles):
            return False
        
        n = len(circles)
        touches_left = [False] * n
        touches_bottom = [False] * n
        touches_right = [False] * n
        touches_top = [False] * n
        
        for i, (x, y, r) in enumerate(circles):
            if x - r <= 0:
                touches_left[i] = True
            if y - r <= 0:
                touches_bottom[i] = True
            if x + r >= xCorner:
                touches_right[i] = True
            if y + r >= yCorner:
                touches_top[i] = True
        
        graph = [[] for _ in range(n)]
        for i in range(n):
            x1, y1, r1 = circles[i]
            for j in range(i + 1, n):
                x2, y2, r2 = circles[j]
                distance_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2
                if distance_sq <= (r1 + r2) ** 2:
                    graph[i].append(j)
                    graph[j].append(i)
        
        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                component = []
                queue = deque([i])
                visited[i] = True
                while queue:
                    node = queue.popleft()
                    component.append(node)
                    for neighbor in graph[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                
                has_left = any(touches_left[node] for node in component)
                has_bottom = any(touches_bottom[node] for node in component)
                has_right = any(touches_right[node] for node in component)
                has_top = any(touches_top[node] for node in component)
                
                if (has_left and has_right) or (has_bottom and has_top):
                    return False
                if (has_left and has_top) and (has_bottom and has_right):
                    return False
        return True