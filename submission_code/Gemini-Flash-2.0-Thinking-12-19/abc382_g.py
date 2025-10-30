import math

def get_tile(x, y, K):
    i = math.floor(x / K)
    j = math.floor(y / K)
    if (i % 2) == (j % 2):
        k = math.floor(y % K)
    else:
        k = math.floor(x % K)
    return (i, j, k)

def get_neighbors(tile, K):
    i, j, k = tile
    neighbors = []
    if (i % 2) == (j % 2): # case 1
        neighbors.append((i + 1, j, 0))
        neighbors.append((i - 1, j, K - 1))
        if 0 <= k < K - 1:
            neighbors.append((i, j, k + 1))
        if k > 0:
            neighbors.append((i, j, k - 1))
        neighbors.append((i, j - 1, 0))
        neighbors.append((i, j + 1, K - 1))
    else: # case 2
        neighbors.append((i, j, (k + 1) % K))
        if k > 0:
            neighbors.append((i, j, (k - 1) % K))
        neighbors.append((i + 1, j, K - 1))
        neighbors.append((i - 1, j, 0))
        neighbors.append((i, j + 1, 0))
        neighbors.append((i, j - 1, K - 1))
    return neighbors

def solve():
    k_val, sx, sy, tx, ty = input().split()
    K = int(k_val)
    Sx = int(sx)
    Sy = int(sy)
    Tx = int(tx)
    Ty = int(ty)
    start_tile = get_tile(Sx + 0.5, Sy + 0.5, K)
    target_tile = get_tile(Tx + 0.5, Ty + 0.5, K)
    if start_tile == target_tile:
        return 0
    queue = [(start_tile, 0)]
    visited = {start_tile}
    while queue:
        current_tile, distance = queue.pop(0)
        if current_tile == target_tile:
            return distance
        for neighbor_tile in get_neighbors(current_tile, K):
            if neighbor_tile not in visited:
                visited.add(neighbor_tile)
                queue.append((neighbor_tile, distance + 1))
    return -1

T = int(input())
results = []
for _ in range(T):
    results.append(solve())

for result in results:
    print(result)