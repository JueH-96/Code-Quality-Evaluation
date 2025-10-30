from typing import List
from collections import deque

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        n = len(circles)
        for x, y, r in circles:
            if x * x + y * y <= r * r:
                return False
            dx = x - xCorner
            dy = y - yCorner
            if dx * dx + dy * dy <= r * r:
                return False
        
        if n == 0:
            return True
        
        left_touch = [False] * n
        right_touch = [False] * n
        bottom_touch = [False] * n
        top_touch = [False] * n
        
        for idx, (x, y, r) in enumerate(circles):
            if x <= r:
                left_touch[idx] = True
            if xCorner - x <= r:
                right_touch[idx] = True
            if y <= r:
                bottom_touch[idx] = True
            if yCorner - y <= r:
                top_touch[idx] = True
        
        graph = [[] for _ in range(n)]
        for i in range(n):
            x1, y1, r1 = circles[i]
            for j in range(i + 1, n):
                x2, y2, r2 = circles[j]
                dx = x1 - x2
                dy = y1 - y2
                d_sq = dx * dx + dy * dy
                if d_sq <= (r1 + r2) ** 2:
                    graph[i].append(j)
                    graph[j].append(i)
        
        visited = [False] * n
        for i in range(n):
            if not visited[i]:
                component_nodes = []
                queue = deque([i])
                visited[i] = True
                while queue:
                    node = queue.popleft()
                    component_nodes.append(node)
                    for neighbor in graph[node]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                
                has_left = any(left_touch[node] for node in component_nodes)
                has_right = any(right_touch[node] for node in component_nodes)
                has_bottom = any(bottom_touch[node] for node in component_nodes)
                has_top = any(top_touch[node] for node in component_nodes)
                
                if (has_left and has_right) or (has_bottom and has_top):
                    return False
        
        return True