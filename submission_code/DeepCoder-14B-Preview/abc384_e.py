import heapq

def main():
    H, W, X = map(int, input().split())
    P, Q = map(int, input().split())
    # Convert to 0-based indices
    P -= 1
    Q -= 1

    grid = []
    for _ in range(H):
        row = list(map(int, input().split()))
        grid.append(row)

    current_strength = grid[P][Q]
    absorbed = set()
    absorbed.add((P, Q))

    # Directions: up, down, left, right
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    heap = []
    # Initialize the heap with adjacent cells
    for dx, dy in dirs:
        nx = P + dx
        ny = Q + dy
        if 0 <= nx < H and 0 <= ny < W:
            heapq.heappush(heap, (-grid[nx][ny], nx, ny))

    while heap:
        s_neg, i, j = heapq.heappop(heap)
        s = -s_neg
        if (i, j) in absorbed:
            continue
        if s < current_strength / X:
            current_strength += s
            absorbed.add((i, j))
            # Add adjacent cells to the heap
            for dx, dy in dirs:
                ni = i + dx
                nj = j + dy
                if 0 <= ni < H and 0 <= nj < W:
                    if (ni, nj) not in absorbed:
                        heapq.heappush(heap, (-grid[ni][nj], ni, nj))
        # Else, do nothing and do not add back to the heap

    print(current_strength)

if __name__ == "__main__":
    main()