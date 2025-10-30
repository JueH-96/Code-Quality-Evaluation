import heapq

def main():
    H, W, T = map(int, input().split())
    grid = []
    start = None
    goal = None
    candies = {}  # (i,j) -> index
    candy_list = []

    for i in range(H):
        line = input().strip()
        grid.append(line)
        for j in range(W):
            c = line[j]
            if c == 'S':
                start = (i, j)
            elif c == 'G':
                goal = (i, j)
            elif c == 'o':
                candies[(i, j)] = len(candy_list)
                candy_list.append((i, j))

    num_candies = len(candy_list)
    steps = [[dict() for _ in range(W)] for _ in range(H)]
    si, sj = start
    initial_mask = 0
    steps[si][sj][initial_mask] = 0
    heap = []
    heapq.heappush(heap, (-initial_mask, 0, si, sj, initial_mask))
    max_candies = -1

    while heap:
        current_priority, current_steps, i, j, mask = heapq.heappop(heap)
        stored_steps = steps[i][j].get(mask, float('inf'))
        if current_steps != stored_steps:
            continue  # Outdated entry, skip

        if (i, j) == goal:
            if current_steps <= T:
                cnt = bin(mask).count('1')
                if cnt > max_candies:
                    max_candies = cnt
            continue  # Continue to check other paths

        if current_steps > T:
            continue

        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni = i + dx
            nj = j + dy
            if 0 <= ni < H and 0 <= nj < W:
                cell = grid[ni][nj]
                if cell == '#':
                    continue

                candy_idx = candies.get((ni, nj), -1)
                new_mask = mask
                if candy_idx != -1:
                    if not (mask & (1 << candy_idx)):
                        new_mask = mask | (1 << candy_idx)

                new_steps = current_steps + 1
                if new_steps > T:
                    continue

                # Check if this new state is better
                current_steps_new = steps[ni][nj].get(new_mask, float('inf'))
                if new_steps < current_steps_new:
                    steps[ni][nj][new_mask] = new_steps
                    heapq.heappush(heap, (-new_mask, new_steps, ni, nj, new_mask))

    print(max_candies if max_candies != -1 else -1)

if __name__ == "__main__":
    main()