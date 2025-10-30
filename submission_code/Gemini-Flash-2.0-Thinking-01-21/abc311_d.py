import collections

def solve():
    n, m = map(int, input().split())
    grid_str = [input() for _ in range(n)]
    grid = [list(row) for row in grid_str]
    
    reachable_ice_squares = set()
    start_pos = (1, 1) # 0-indexed (1, 1) is (2, 2) in 1-indexed
    if grid[start_pos[0]][start_pos[1]] == '#':
        print(0)
        return
        
    reachable_ice_squares.add(start_pos)
    queue = collections.deque([start_pos])
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # up, down, left, right
    
    while queue:
        current_pos = queue.popleft()
        r, c = current_pos
        
        for dr, dc in directions:
            last_ice_pos = None
            path_squares = []
            next_r, next_c = r + dr, c + dc
            while 0 <= next_r < n and 0 <= next_c < m and grid[next_r][next_c] == '.':
                path_squares.append((next_r, next_c))
                last_ice_pos = (next_r, next_c)
                next_r += dr
                next_c += dc
                
            if last_ice_pos:
                final_r, final_c = last_ice_pos
                for sq in path_squares:
                    if sq not in reachable_ice_squares:
                        reachable_ice_squares.add(sq)
                        queue.append(sq)
                if last_ice_pos not in reachable_ice_squares:
                    reachable_ice_squares.add(last_ice_pos)
                    queue.append(last_ice_pos)
                    
    print(len(reachable_ice_squares))

if __name__ == '__main__':
    solve()