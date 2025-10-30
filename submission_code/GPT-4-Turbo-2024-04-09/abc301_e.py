from sys import stdin, stdout
from collections import deque

def read_input():
    import sys
    input = sys.stdin.read
    data = input().split()
    H = int(data[0])
    W = int(data[1])
    T = int(data[2])
    grid = [data[i + 3] for i in range(H)]
    return H, W, T, grid

def find_positions(grid, H, W):
    start = goal = None
    candies = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 'S':
                start = (r, c)
            elif grid[r][c] == 'G':
                goal = (r, c)
            elif grid[r][c] == 'o':
                candies.append((r, c))
    return start, goal, candies

def bfs_min_distance(grid, H, W, start, goal):
    queue = deque([start])
    distances = [[float('inf')] * W for _ in range(H)]
    sr, sc = start
    distances[sr][sc] = 0
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while queue:
        r, c = queue.popleft()
        current_distance = distances[r][c]
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] != '#':
                if distances[nr][nc] > current_distance + 1:
                    distances[nr][nc] = current_distance + 1
                    queue.append((nr, nc))
    
    return distances[goal[0]][goal[1]]

def solve():
    H, W, T, grid = read_input()
    start, goal, candies = find_positions(grid, H, W)
    
    # First check if it's possible to reach the goal within T moves
    min_moves_to_goal = bfs_min_distance(grid, H, W, start, goal)
    if min_moves_to_goal > T:
        print(-1)
        return
    
    # If there are no candies, just return 0 if we can reach the goal
    if not candies:
        print(0)
        return
    
    # Use bitmask to represent visited candies
    from itertools import combinations
    
    max_candies = 0
    candy_count = len(candies)
    all_positions = [start] + candies + [goal]
    
    # Precompute distances between all important points (start, candies, goal)
    distances = [[None] * (2 + candy_count) for _ in range(2 + candy_count)]
    for i, pos1 in enumerate(all_positions):
        distances[i][i] = 0
        for j, pos2 in enumerate(all_positions[i+1:], i+1):
            dist = bfs_min_distance(grid, H, W, pos1, pos2)
            distances[i][j] = dist
            distances[j][i] = dist
    
    # Try all combinations of candies
    for num_candies in range(1, candy_count + 1):
        for candy_indices in combinations(range(candy_count), num_candies):
            # Check all permutations of visiting these candies
            from itertools import permutations
            for perm in permutations(candy_indices):
                total_moves = distances[0][1 + perm[0]]  # Start to first candy
                for k in range(len(perm) - 1):
                    total_moves += distances[1 + perm[k]][1 + perm[k + 1]]
                total_moves += distances[1 + perm[-1]][1 + candy_count]  # Last candy to goal
                
                if total_moves <= T:
                    max_candies = max(max_candies, num_candies)
    
    print(max_candies)

solve()