import collections

def solve():
    n, m = map(int, input().split())
    grid_str = [input() for _ in range(n)]
    grid = [list(row) for row in grid_str]
    
    reachable_ice_squares = set()
    start_pos = (1, 1)
    reachable_ice_squares.add(start_pos)
    queue = collections.deque([start_pos])
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)] # up, down, left, right
    
    while queue:
        current_pos = queue.popleft()
        r, c = current_pos
        
        for dr, dc in directions:
            path_taken = []
            current_r, current_c = r, c
            while True:
                next_r, next_c = current_r + dr, current_c + dc
                if 0 <= next_r < n and 0 <= next_c < m and grid[next_r][next_c] == '.':
                    path_taken.append((next_r, next_c))
                    current_r, current_c = next_r, next_c
                else:
                    break
            
            if path_taken:
                final_position = path_taken[-1]
                for square in path_taken:
                    if square not in reachable_ice_squares:
                        reachable_ice_squares.add(square)
                        queue.append(square)
                if final_position not in reachable_ice_squares:
                    reachable_ice_squares.add(final_position)
                    queue.append(final_position)
                    
    print(len(reachable_ice_squares))

if __name__ == '__main__':
    solve()