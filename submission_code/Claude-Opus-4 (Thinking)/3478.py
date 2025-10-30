class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: List[List[int]]) -> bool:
        from collections import deque
        
        # Check if a circle intersects with the rectangle
        def intersects_rectangle(circle):
            x, y, r = circle
            # Find closest point on rectangle to circle center
            closest_x = max(0, min(x, xCorner))
            closest_y = max(0, min(y, yCorner))
            # Check if this closest point is within the circle
            dist_sq = (x - closest_x) ** 2 + (y - closest_y) ** 2
            return dist_sq <= r ** 2
        
        # Filter out circles that don't intersect with the rectangle
        circles = [c for c in circles if intersects_rectangle(c)]
        n = len(circles)
        
        if n == 0:
            return True
        
        # Check if a point is inside or on a circle
        def point_in_circle(px, py, circle):
            x, y, r = circle
            dist_sq = (px - x) ** 2 + (py - y) ** 2
            return dist_sq <= r ** 2
        
        # Check if starting or ending corner is blocked
        for circle in circles:
            if point_in_circle(0, 0, circle) or point_in_circle(xCorner, yCorner, circle):
                return False
        
        # Check if a circle touches each edge
        def touches_left(circle):
            x, y, r = circle
            return x - r <= 0
        
        def touches_right(circle):
            x, y, r = circle
            return x + r >= xCorner
        
        def touches_bottom(circle):
            x, y, r = circle
            return y - r <= 0
        
        def touches_top(circle):
            x, y, r = circle
            return y + r >= yCorner
        
        # Check if two circles overlap or touch
        def circles_touch(c1, c2):
            x1, y1, r1 = c1
            x2, y2, r2 = c2
            dist_sq = (x1 - x2) ** 2 + (y1 - y2) ** 2
            return dist_sq <= (r1 + r2) ** 2
        
        # Build adjacency list for circles
        adj = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if circles_touch(circles[i], circles[j]):
                    adj[i].append(j)
                    adj[j].append(i)
        
        # Find circles touching each edge
        left_circles = []
        right_circles = []
        bottom_circles = []
        top_circles = []
        
        for i in range(n):
            if touches_left(circles[i]):
                left_circles.append(i)
            if touches_right(circles[i]):
                right_circles.append(i)
            if touches_bottom(circles[i]):
                bottom_circles.append(i)
            if touches_top(circles[i]):
                top_circles.append(i)
        
        # BFS to check if there's a path from sources to targets
        def has_path(sources, targets):
            if not sources or not targets:
                return False
            
            target_set = set(targets)
            visited = [False] * n
            queue = deque(sources)
            for s in sources:
                visited[s] = True
            
            while queue:
                curr = queue.popleft()
                if curr in target_set:
                    return True
                
                for neighbor in adj[curr]:
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(neighbor)
            
            return False
        
        # Check if left edge connects to top edge
        if has_path(left_circles, top_circles):
            return False
        
        # Check if bottom edge connects to right edge  
        if has_path(bottom_circles, right_circles):
            return False
        
        return True