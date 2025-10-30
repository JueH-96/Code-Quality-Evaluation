import collections

def solve():
    h, w, t = map(int, input().split())
    grid = [list(input()) for _ in range(h)]
    start_pos = None
    goal_pos = None
    candy_positions = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 'S':
                start_pos = (r, c)
            elif grid[r][c] == 'G':
                goal_pos = (r, c)
            elif grid[r][c] == 'o':
                candy_positions.append((r, c))
                
    candy_indices = {pos: i for i, pos in enumerate(candy_positions)}
    num_candies = len(candy_positions)
    max_candy_mask_val = 1 << num_candies
    min_moves = {}
    
    start_r, start_c = start_pos
    goal_r, goal_c = goal_pos
    
    initial_mask = 0
    if grid[start_r][start_c] == 'o':
        candy_index = candy_indices.get(start_pos)
        if candy_index is not None:
            initial_mask = (1 << candy_index)
            
    min_moves[(start_r, start_c, initial_mask)] = 0
    queue = collections.deque([(start_r, start_c, initial_mask, 0)])
    max_candies_collected = -1
    
    while queue:
        r, c, mask, moves = queue.popleft()
        if moves > t:
            continue
        if (r, c) == goal_pos:
            candies_count = bin(mask).count('1')
            max_candies_collected = max(max_candies_collected, candies_count)
            
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != '#':
                new_mask = mask
                if grid[nr][nc] == 'o':
                    candy_index = candy_indices.get((nr, nc))
                    if candy_index is not None:
                        new_mask = mask | (1 << candy_index)
                        
                next_state = (nr, nc, new_mask)
                if next_state not in min_moves or moves + 1 < min_moves[next_state]:
                    min_moves[next_state] = moves + 1
                    queue.append((nr, nc, new_mask, moves + 1))
                    
    print(max_candies_collected)

if __name__ == '__main__':
    solve()