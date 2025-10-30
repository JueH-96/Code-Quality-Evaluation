import sys
sys.setrecursionlimit(1000000)

# Read input
data = sys.stdin.read().split()
index = 0
N = int(data[index])
index += 1
H = int(data[index])
index += 1
W = int(data[index])
index += 1
A = [0] * N
B = [0] * N
for i in range(N):
    A[i] = int(data[index])
    index += 1
    B[i] = int(data[index])
    index += 1

total_area = H * W

# Precompute all possible placements for each tile
placements = []
for i in range(N):
    pl = []
    a1, b1 = A[i], B[i]
    # Orientation A x B
    for row in range(H - a1 + 1):
        for col in range(W - b1 + 1):
            cells = frozenset([(row + di) * W + (col + dj) for di in range(a1) for dj in range(b1)])
            pl.append(cells)
    if a1 != b1:  # Add B x A orientation if not square
        a2, b2 = b1, a1
        for row in range(H - a2 + 1):
            for col in range(W - b2 + 1):
                cells = frozenset([(row + di) * W + (col + dj) for di in range(a2) for dj in range(b2)])
                pl.append(cells)
    placements.append(pl)

# Backtracking function to check if tiles can be placed
def backtrack(index, tiles_subset, occupied):
    if index == len(tiles_subset):
        return True
    tile_idx = tiles_subset[index]
    for placement_set in placements[tile_idx]:
        if placement_set.isdisjoint(occupied):  # No overlap
            # Add cells to occupied
            occupied.update(placement_set)
            if backtrack(index + 1, tiles_subset, occupied):
                return True
            # Remove cells from occupied
            for cell in placement_set:
                occupied.remove(cell)
    return False

# Iterate through all subsets using bitmask
for mask in range(1 << N):
    sum_area = 0
    tiles = []
    for i in range(N):
        if mask & (1 << i):
            sum_area += A[i] * B[i]
            tiles.append(i)
    if sum_area == total_area:
        # Sort tiles by area descending
        tiles.sort(key=lambda x: A[x] * B[x], reverse=True)
        # Start with empty occupied set
        occupied = set()
        if backtrack(0, tiles, occupied):
            print("Yes")
            sys.exit()

# If no subset works
print("No")