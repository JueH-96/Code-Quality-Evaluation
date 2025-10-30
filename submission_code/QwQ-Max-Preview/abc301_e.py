import heapq

def main():
    import sys
    input = sys.stdin.read().split()
    H = int(input[0])
    W = int(input[1])
    T = int(input[2])
    grid = input[3:3+H]
    
    # Find S, G, and candies
    start = None
    goal = None
    candies = []
    for i in range(H):
        for j in range(W):
            c = grid[i][j]
            if c == 'S':
                start = (i, j)
            elif c == 'G':
                goal = (i, j)
            elif c == 'o':
                candies.append((i, j))
    
    if not start or not goal:
        print(-1)
        return
    
    m = len(candies)
    # Create a map from (i,j) to candy index
    candy_map = {}
    for idx, (i, j) in enumerate(candies):
        candy_map[(i, j)] = idx
    
    # Check if S and G are same (problem states they are unique, but just in case)
    if start == goal:
        print(0 if T >=0 else -1)
        return
    
    # Dijkstra's algorithm with priority queue (steps, i, j, mask)
    heap = []
    heapq.heappush(heap, (0, start[0], start[1], 0))
    steps = {}
    steps[(start[0], start[1], 0)] = 0
    max_candies = -1
    target = goal
    
    while heap:
        current_steps, i, j, mask = heapq.heappop(heap)
        if (i, j) == target:
            cnt = bin(mask).count('1')
            if cnt > max_candies and current_steps <= T:
                max_candies = cnt
            # Continue processing to find higher counts
            # But if current_steps is minimal, other paths may have higher masks
        if current_steps > T:
            continue
        # Check if this state is already processed with lower steps
        if steps.get((i, j, mask), float('inf')) < current_steps:
            continue
        # Explore all four directions
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni = i + di
            nj = j + dj
            if 0 <= ni < H and 0 <= nj < W:
                if grid[ni][nj] == '#':
                    continue
                new_mask = mask
                if (ni, nj) in candy_map:
                    candy_idx = candy_map[(ni, nj)]
                    new_mask = mask | (1 << candy_idx)
                new_steps = current_steps + 1
                key = (ni, nj, new_mask)
                if new_steps < steps.get(key, float('inf')):
                    steps[key] = new_steps
                    heapq.heappush(heap, (new_steps, ni, nj, new_mask))
    
    # Check if there is a path to G within T steps but with 0 candies
    if max_candies == -1:
        # Check if G is reachable in T steps
        # Re-run BFS to find minimal steps to G without considering candies
        from collections import deque
        visited = [[False]*W for _ in range(H)]
        q = deque()
        q.append( (start[0], start[1], 0) )
        visited[start[0]][start[1]] = True
        found = False
        while q:
            i, j, s = q.popleft()
            if (i, j) == target:
                if s <= T:
                    found = True
                    break
                else:
                    continue
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni = i + di
                nj = j + dj
                if 0 <= ni < H and 0 <= nj < W and not visited[ni][nj] and grid[ni][nj] != '#':
                    visited[ni][nj] = True
                    q.append( (ni, nj, s+1) )
        if found:
            print(0)
        else:
            print(-1)
    else:
        print(max_candies if max_candies != -1 else 0)

if __name__ == '__main__':
    main()