class Solution:
    def canReachCorner(self, xCorner: int, yCorner: int, circles: list[list[int]]) -> bool:
        def dist_sq(p1, p2):
            return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

        def is_inside_circle(point, circle):
            return dist_sq(point, (circle[0], circle[1])) <= circle[2] ** 2

        if any(is_inside_circle((0, 0), circle) for circle in circles) or \
           any(is_inside_circle((xCorner, yCorner), circle) for circle in circles):
            return False

        adj = [[] for _ in range(len(circles))]
        for i in range(len(circles)):
            for j in range(i + 1, len(circles)):
                if dist_sq((circles[i][0], circles[i][1]), (circles[j][0], circles[j][1])) <= (circles[i][2] + circles[j][2]) ** 2:
                    adj[i].append(j)
                    adj[j].append(i)

        visited = [False] * len(circles)
        
        def dfs(u):
            visited[u] = True
            for v in adj[u]:
                if not visited[v]:
                    dfs(v)

        
        blocked_regions = []
        for i in range(len(circles)):
            if not visited[i]:
                region = []
                q = [i]
                visited_region = [False] * len(circles)
                visited_region[i] = True
                while q:
                    curr = q.pop(0)
                    region.append(curr)
                    for neighbor in adj[curr]:
                        if not visited_region[neighbor]:
                            visited_region[neighbor] = True
                            q.append(neighbor)
                
                blocked_regions.append(region)

        for region in blocked_regions:
            blocks_left = False
            blocks_right = False
            blocks_top = False
            blocks_bottom = False
            for circle_idx in region:
                circle = circles[circle_idx]
                if circle[0] - circle[2] <= 0:
                    blocks_left = True
                if circle[0] + circle[2] >= xCorner:
                    blocks_right = True
                if circle[1] - circle[2] <= 0:
                    blocks_bottom = True
                if circle[1] + circle[2] >= yCorner:
                    blocks_top = True
            
            if blocks_left and blocks_top: return False
            if blocks_left and blocks_right: return False
            if blocks_right and blocks_bottom: return False
            if blocks_bottom and blocks_top: return False
        
        return True