import heapq

def main():
    H, W, T = map(int, input().split())
    grid = [input().strip() for _ in range(H)]
    
    # Find positions of S, G, and o's
    S = None
    G = None
    o_pos = []
    for i in range(H):
        for j in range(W):
            c = grid[i][j]
            if c == 'S':
                S = (i, j)
            elif c == 'G':
                G = (i, j)
            elif c == 'o':
                o_pos.append((i, j))
    
    # Assign indices to o's
    o_index = {}
    for idx, (i, j) in enumerate(o_pos):
        o_index[(i, j)] = idx
    K = len(o_pos)
    
    # Precompute distances from all cells to G using BFS
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    def bfs(start_i, start_j):
        dist = [[-1]*W for _ in range(H)]
        q = [(start_i, start_j)]
        dist[start_i][start_j] = 0
        for i, j in q:
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] != '#' and dist[ni][nj] == -1:
                    dist[ni][nj] = dist[i][j] + 1
                    q.append((ni, nj))
        return dist
    dist_g = bfs(G[0], G[1])
    
    # Check if S is unreachable from G
    if dist_g[S[0]][S[1]] == -1 or dist_g[S[0]][S[1]] > T:
        print(-1)
        return
    
    max_candies = -1
    heap = []
    heapq.heappush(heap, (0, S[0], S[1], 0))
    visited = {}
    visited_key = (S[0], S[1], 0)
    visited[visited_key] = 0
    
    while heap:
        steps, i, j, mask = heapq.heappop(heap)
        if visited.get((i, j, mask), float('inf')) < steps:
            continue
        
        # Check if current state can reach G within T steps
        if steps + dist_g[i][j] <= T:
            current = bin(mask).count('1')
            if current > max_candies:
                max_candies = current
        
        for di, dj in dirs:
            ni, nj = i + di, j + dj
            if 0 <= ni < H and 0 <= nj < W and grid[ni][nj] != '#':
                new_steps = steps + 1
                if new_steps > T:
                    continue
                if new_steps + dist_g[ni][nj] > T:
                    continue
                
                new_mask = mask
                if (ni, nj) in o_index:
                    c = o_index[(ni, nj)]
                    if not (mask & (1 << c)):
                        new_mask = mask | (1 << c)
                
                key = (ni, nj, new_mask)
                if visited.get(key, float('inf')) > new_steps:
                    visited[key] = new_steps
                    heapq.heappush(heap, (new_steps, ni, nj, new_mask))
    
    print(max_candies if max_candies != -1 else -1)

if __name__ == "__main__":
    main()