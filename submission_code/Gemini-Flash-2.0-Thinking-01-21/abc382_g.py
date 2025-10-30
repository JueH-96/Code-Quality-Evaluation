import collections

def solve():
    k, sx, sy, tx, ty = map(int, input().split())
    start_point = (sx + 0.5, sy + 0.5)
    target_point = (tx + 0.5, ty + 0.5)
    
    def get_tile_index(point):
        x, y = point
        i = int(x // k)
        j = int(y // k)
        if (i + j) % 2 == 0:
            tile_type = 1
            kt = int(y - j * k)
            return (i, j, kt)
        else:
            tile_type = 2
            kt = int(x - i * k)
            return (i, j, kt)
            
    start_tile = get_tile_index(start_point)
    target_tile = get_tile_index(target_point)
    
    if start_tile == target_tile:
        print(0)
        return
        
    def get_neighbors(tile):
        i, j, kk = tile
        neighbors = []
        if (i + j) % 2 == 0: # type 1
            potential_neighbors = [
                (i + 1, j, kk), (i - 1, j, kk), (i, j, kk + 1), (i, j, kk - 1),
                (i, j + 1, -1), (i, j - 1, -1), (i + 1, j, 0), (i - 1, j, k - 1)
            ]
            for ni, nj, nk in potential_neighbors:
                if 0 <= nk < k:
                    neighbors.append((ni, nj, nk))
                elif nk == -1:
                    for k_prime in range(k):
                        neighbors.append((ni, nj, k_prime))
                        
        else: # type 2
            potential_neighbors = [
                (i, j + 1, kk), (i, j - 1, kk), (i, j, kk + 1), (i, j, kk - 1),
                (i + 1, j, -1), (i - 1, j, -1), (i, j + 1, 0), (i, j - 1, k - 1)
            ]
            for ni, nj, nk in potential_neighbors:
                if 0 <= nk < k:
                    neighbors.append((ni, nj, nk))
                elif nk == -1:
                    for k_prime in range(k):
                        neighbors.append((ni, nj, k_prime))
                        
        return neighbors

    queue = collections.deque([(start_tile, 0)])
    visited = {start_tile}
    
    while queue:
        current_tile, distance = queue.popleft()
        if current_tile == target_tile:
            print(distance)
            return
            
        for neighbor_tile in get_neighbors(current_tile):
            if neighbor_tile not in visited:
                visited.add(neighbor_tile)
                queue.append((neighbor_tile, distance + 1))
                
    print("Target not reachable? Should not happen.")

t = int(input())
for _ in range(t):
    solve()