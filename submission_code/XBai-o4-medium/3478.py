from typing import List
from collections import deque

class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        # Step 1: Check if start (0,0) or end (xCorner, yCorner) is inside any circle
        for circle in circles:
            x, y, r = circle
            # Check if (0, 0) is inside or on the circle
            if x * x + y * y <= r * r:
                return False
            # Check if (xCorner, yCorner) is inside or on the circle
            dx = xCorner - x
            dy = yCorner - y
            if dx * dx + dy * dy <= r * r:
                return False
        
        # Build adjacency list for circles that overlap
        n = len(circles)
        adj = [[] for _ in range(n)]
        for i in range(n):
            x1, y1, r1 = circles[i]
            for j in range(i + 1, n):
                x2, y2, r2 = circles[j]
                dx = x1 - x2
                dy = y1 - y2
                dist_sq = dx * dx + dy * dy
                r_sum = r1 + r2
                if dist_sq <= r_sum * r_sum:
                    adj[i].append(j)
                    adj[j].append(i)
        
        # Helper function to check connection using BFS
        def check_connection(is_start, is_end):
            visited = set()
            q = deque()
            for i in range(n):
                if is_start(i):
                    q.append(i)
                    visited.add(i)
            while q:
                current = q.popleft()
                if is_end(current):
                    return True
                for neighbor in adj[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        q.append(neighbor)
            return False
        
        # Check for left-right connection
        left_right = check_connection(
            lambda i: (circles[i][0] - circles[i][2] <= 0),
            lambda i: (circles[i][0] + circles[i][2] >= xCorner)
        )
        
        # Check for bottom-top connection
        bottom_top = check_connection(
            lambda i: (circles[i][1] - circles[i][2] <= 0),
            lambda i: (circles[i][1] + circles[i][2] >= yCorner)
        )
        
        return not (left_right or bottom_top)