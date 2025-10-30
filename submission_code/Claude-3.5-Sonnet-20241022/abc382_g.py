def find_tile(x, y, K):
    # Find i, j coordinates of the grid cell containing (x, y)
    i = x // K
    j = y // K
    
    # Adjust for negative coordinates
    if x < 0 and x % K != 0:
        i -= 1
    if y < 0 and y % K != 0:
        j -= 1
    
    # Find k based on position within the cell
    if i % 2 == j % 2:  # Same parity - horizontal tile
        k = y - (j * K)
        if k >= K:
            k = K - 1
        elif k < 0:
            k = 0
    else:  # Different parity - vertical tile
        k = x - (i * K)
        if k >= K:
            k = K - 1
        elif k < 0:
            k = 0
            
    return (i, j, k)

def are_adjacent(tile1, tile2, K):
    i1, j1, k1 = tile1
    i2, j2, k2 = tile2
    
    # Same cell
    if i1 == i2 and j1 == j2:
        return abs(k1 - k2) == 1
    
    # Adjacent cells
    if abs(i1 - i2) + abs(j1 - j2) == 1:
        # For horizontal tiles (same parity)
        if i1 % 2 == j1 % 2 and i2 % 2 == j2 % 2:
            return k1 == k2
        # For vertical tiles (different parity)
        elif i1 % 2 != j1 % 2 and i2 % 2 != j2 % 2:
            return k1 == k2
        # Between horizontal and vertical
        else:
            if i1 % 2 == j1 % 2:  # tile1 is horizontal
                return k2 == i2 * K + k1 - i1 * K
            else:  # tile1 is vertical
                return k1 == i1 * K + k2 - i2 * K
    
    return False

def solve(K, Sx, Sy, Tx, Ty):
    # If start and target are in same tile
    start_tile = find_tile(Sx, Sy, K)
    target_tile = find_tile(Tx, Ty, K)
    
    if start_tile == target_tile:
        return 0
    
    # Manhattan distance approximation
    dx = abs(Tx - Sx)
    dy = abs(Ty - Sy)
    
    # For very large distances, we can use an approximation
    if max(dx, dy) > K * 10:
        return (dx + dy) // K
    
    # BFS for shorter distances
    from collections import deque
    
    queue = deque([(start_tile, 0)])
    visited = {start_tile}
    
    while queue:
        current_tile, dist = queue.popleft()
        
        if current_tile == target_tile:
            return dist
        
        i, j, k = current_tile
        
        # Try all possible adjacent tiles
        for ni, nj in [(i+1,j), (i-1,j), (i,j+1), (i,j-1)]:
            for nk in range(K):
                next_tile = (ni, nj, nk)
                if next_tile not in visited and are_adjacent(current_tile, next_tile, K):
                    if next_tile == target_tile:
                        return dist + 1
                    visited.add(next_tile)
                    queue.append((next_tile, dist + 1))
    
    return -1  # Should not reach here

def main():
    T = int(input())
    for _ in range(T):
        K, Sx, Sy, Tx, Ty = map(int, input().split())
        print(solve(K, Sx, Sy, Tx, Ty))

if __name__ == "__main__":
    main()