import collections

def solve():
    sx, sy = map(int, input().split())
    tx, ty = map(int, input().split())
    start_x = sx + 0.5
    start_y = sy + 0.5
    target_x = tx + 0.5
    target_y = ty + 0.5
    
    def get_tile_index(x, y):
        i = int(x)
        j = int(y)
        return i, j
        
    def get_tile_type(i, j):
        return (i + j) % 2
        
    def get_tile_boundaries(i, j):
        if get_tile_type(i, j) == 0: # i+j is even, type 0
            return [(i, i + 2), (j, j + 1)]
        else: # i+j is odd, type 1
            return [(i - 1, i + 1), (j, j + 1)]
            
    start_tile_index = get_tile_index(start_x, start_y)
    target_tile_index = get_tile_index(target_x, target_y)
    
    if start_tile_index == target_tile_index:
        print(0)
        return
        
    start_i, start_j = start_tile_index
    target_i, target_j = target_tile_index
    
    queue = collections.deque([(start_i, start_j, 0)])
    visited_tiles = set([(start_i, start_j)])
    distance = {(start_i, start_j): 0}
    
    while queue:
        current_i, current_j, current_cost = queue.popleft()
        
        if (current_i, current_j) == (target_i, target_j):
            print(current_cost)
            return
            
        # Possible moves to adjacent tiles
        
        # Move right
        next_i_right = current_i + 1
        if (next_i_right, current_j) not in visited_tiles:
            visited_tiles.add((next_i_right, current_j))
            distance[(next_i_right, current_j)] = current_cost + 1
            queue.append((next_i_right, current_j, current_cost + 1))
            
        # Move left
        next_i_left = current_i - 1
        if (next_i_left, current_j) not in visited_tiles:
            visited_tiles.add((next_i_left, current_j))
            distance[(next_i_left, current_j)] = current_cost + 1
            queue.append((next_i_left, current_j, current_cost + 1))
            
        # Move up
        next_j_up = current_j + 1
        if (current_i, next_j_up) not in visited_tiles:
            visited_tiles.add((current_i, next_j_up))
            distance[(current_i, next_j_up)] = current_cost + 1
            queue.append((current_i, next_j_up, current_cost + 1))

        # Move down
        next_j_down = current_j - 1
        if (current_i, next_j_down) not in visited_tiles:
            visited_tiles.add((current_i, next_j_down))
            distance[(current_i, next_j_down)] = current_cost + 1
            queue.append((current_i, next_j_down, current_cost + 1))

    print("Error: Target not reachable")

if __name__ == '__main__':
    solve()