import heapq

def solve():
    h, w, t = map(int, input().split())
    grid = [input() for _ in range(h)]
    start_pos = None
    goal_pos = None
    candy_locations = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 'S':
                start_pos = (r, c)
            elif grid[r][c] == 'G':
                goal_pos = (r, c)
            elif grid[r][c] == 'o':
                candy_locations.append((r, c))
                
    num_candies = len(candy_locations)
    min_moves = {}
    for r in range(h):
        for c in range(w):
            for mask in range(1 << num_candies):
                min_moves[(r, c, mask)] = float('inf')
                
    min_moves[start_pos + (0,)] = 0
    pq = [(0, start_pos[0], start_pos[1], 0)]
    
    while pq:
        moves, r, c, mask = heapq.heappop(pq)
        if moves > t:
            continue
        if moves > min_moves[(r, c, mask)]:
            continue
            
        if (r, c) == goal_pos:
            pass 
            
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != '#':
                next_moves = moves + 1
                if next_moves > t:
                    continue
                next_mask = mask
                is_candy = False
                candy_index = -1
                for i in range(num_candies):
                    if candy_locations[i] == (nr, nc):
                        is_candy = True
                        candy_index = i
                        break
                if is_candy:
                    if not (mask & (1 << candy_index)):
                        next_mask = mask | (1 << candy_index)
                        
                if next_moves < min_moves[(nr, nc, next_mask)]:
                    min_moves[(nr, nc, next_mask)] = next_moves
                    heapq.heappush(pq, (next_moves, nr, nc, next_mask))
                    
    max_collected_candies = -1
    for mask in range(1 << num_candies):
        if min_moves[(goal_pos[0], goal_pos[1], mask)] <= t:
            collected_candies = bin(mask).count('1')
            max_collected_candies = max(max_collected_candies, collected_candies)
            
    print(max_collected_candies)

if __name__ == '__main__':
    solve()