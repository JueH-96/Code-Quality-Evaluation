from itertools import product

def count_possible_paths(H, W, grid, K):
    """
    Computes the number of possible paths of length K on a grid HxW, avoiding blocked cells (. and #).
    A path cannot revisit any cell.
    """
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def legal(i, j, k):
        return 0 <= i < H and 0 <= j < W and grid[i][j] != "#" and not (i, j) in legal.visited[k]

    def next_path(i, j, k):
        for di, dj in moves:
            ni, nj = i + di, j + dj
            if legal(ni, nj, k + 1):
                yield ni, nj
                legal.visited[k + 1].add((i, j))

    def generate_paths(k):
        for ij in legal_coordinates:
            prev = {(ij)}
            yield [ij]
            for l in range(k):
                next = set()
                legal.visited[l + 1] = set()
                for ij in prev:
                    next |= set(next_path(*ij, l))
                prev = next
                for ij in next:
                    yield [*paths[l], ij]

    legal.visited = [set() for _ in range(K)]
    start = {i for i in product(range(H), range(W)) if grid[i[0]][i[1]] != "#"} 
    legal_coordinates = [i for i in product(range(H), range(W)) if grid[i[0]][i[1]] != "#" and i not in start]
    paths = list(generate_paths(K))
    return len(paths)

# Read inputs
H, W, K = map(int, input().split())
grid = [input() for _ in range(H)]

# Solve and print the result
print(count_possible_paths(H, W, grid, K))