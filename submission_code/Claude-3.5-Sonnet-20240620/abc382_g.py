# YOUR CODE HERE
def find_tile(x, y, K):
    i, j = x // K, y // K
    if (i + j) % 2 == 0:
        k = y % K
    else:
        k = x % K
    return i, j, k

def are_adjacent(tile1, tile2):
    i1, j1, k1 = tile1
    i2, j2, k2 = tile2
    if i1 == i2 and j1 == j2:
        return abs(k1 - k2) == 1
    if abs(i1 - i2) + abs(j1 - j2) == 1:
        return (i1 + j1) % 2 != (i2 + j2) % 2 or k1 == k2
    return False

def min_moves(K, Sx, Sy, Tx, Ty):
    start_tile = find_tile(Sx, Sy, K)
    end_tile = find_tile(Tx, Ty, K)
    
    if start_tile == end_tile:
        return 0
    
    queue = [(start_tile, 0)]
    visited = set([start_tile])
    
    while queue:
        current_tile, moves = queue.pop(0)
        
        if current_tile == end_tile:
            return moves
        
        i, j, k = current_tile
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                if di == 0 and dj == 0:
                    continue
                for dk in range(K):
                    next_tile = (i + di, j + dj, dk)
                    if next_tile not in visited and are_adjacent(current_tile, next_tile):
                        queue.append((next_tile, moves + 1))
                        visited.add(next_tile)
    
    return -1  # Should never reach here if input is valid

T = int(input())
for _ in range(T):
    K, Sx, Sy, Tx, Ty = map(int, input().split())
    print(min_moves(K, Sx, Sy, Tx, Ty))