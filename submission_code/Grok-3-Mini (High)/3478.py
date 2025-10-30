import collections
from typing import List

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        # Check if start or end is inside or on any circle
        for x, y, r in circles:
            if (x * x + y * y) <= r * r:  # Distance to (0, 0)
                return False
            dx = x - xCorner
            dy = y - yCorner
            if (dx * dx + dy * dy) <= r * r:  # Distance to (xCorner, yCorner)
                return False
        
        num_circles = len(circles)
        total_nodes = num_circles + 4  # 0 to num_circles-1: circles, num_circles: left, num_circles+1: right, num_circles+2: bottom, num_circles+3: top
        adj = [[] for _ in range(total_nodes)]
        
        # Add edges between circles if they intersect or touch
        for i in range(num_circles):
            for j in range(i + 1, num_circles):
                x1, y1, r1 = circles[i]
                x2, y2, r2 = circles[j]
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx * dx + dy * dy
                sum_r = r1 + r2
                sum_r_sq = sum_r * sum_r
                if dist_sq <= sum_r_sq:
                    adj[i].append(j)
                    adj[j].append(i)
        
        # Define wall node indices
        left_idx = num_circles
        right_idx = num_circles + 1
        bottom_idx = num_circles + 2
        top_idx = num_circles + 3
        
        # Add edges between circles and wall nodes if they touch or intersect the walls
        for i in range(num_circles):
            x, y, r = circles[i]
            if x - r <= 0:  # Touches or intersects left wall
                adj[i].append(left_idx)
                adj[left_idx].append(i)
            if x + r >= xCorner:  # Touches or intersects right wall
                adj[i].append(right_idx)
                adj[right_idx].append(i)
            if y - r <= 0:  # Touches or intersects bottom wall
                adj[i].append(bottom_idx)
                adj[bottom_idx].append(i)
            if y + r >= yCorner:  # Touches or intersects top wall
                adj[i].append(top_idx)
                adj[top_idx].append(i)
        
        # Define BFS function to check if there is a path from start to target
        def bfs(start, target):
            if start == target:
                return True  # Though unlikely in this setup, handle for completeness
            visited = [False] * total_nodes
            queue = collections.deque([start])
            visited[start] = True
            while queue:
                node = queue.popleft()
                if node == target:
                    return True
                for neighbor in adj[node]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            return False
        
        # Check if left can reach right or bottom can reach top
        if bfs(left_idx, right_idx) or bfs(bottom_idx, top_idx):
            return False
        
        return True