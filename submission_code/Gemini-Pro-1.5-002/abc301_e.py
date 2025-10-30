# YOUR CODE HERE
import heapq

def solve():
    H, W, T = map(int, input().split())
    grid = [input() for _ in range(H)]
    
    start = None
    goal = None
    candies = []
    
    for i in range(H):
        for j in range(W):
            if grid[i][j] == 'S':
                start = (i, j)
            elif grid[i][j] == 'G':
                goal = (i, j)
            elif grid[i][j] == 'o':
                candies.append((i, j))
    
    num_candies = len(candies)
    
    def bfs(start_node, end_node):
        q = [(0, start_node)]
        visited = set()
        min_dist = float('inf')
        
        while q:
            dist, current = heapq.heappop(q)
            
            if current == end_node:
                min_dist = min(min_dist, dist)
                continue
                
            if current in visited:
                continue
            visited.add(current)
            
            r, c = current
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#':
                    heapq.heappush(q, (dist + 1, (nr, nc)))
        
        return min_dist if min_dist != float('inf') else -1

    max_collected_candies = -1
    for i in range(1 << num_candies):
        collected_candies = 0
        current_path_time = 0
        current_path = []
        
        current_path.append(start)
        for j in range(num_candies):
            if (i >> j) & 1:
                collected_candies += 1
                current_path.append(candies[j])
        current_path.append(goal)

        for k in range(len(current_path) - 1):
            dist = bfs(current_path[k], current_path[k+1])
            if dist == -1:
                current_path_time = float('inf')
                break
            current_path_time += dist

        if current_path_time <= T:
            max_collected_candies = max(max_collected_candies, collected_candies)

    print(max_collected_candies)

solve()