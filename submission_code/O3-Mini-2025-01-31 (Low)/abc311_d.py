def main():
    import sys
    from collections import deque
    data = sys.stdin.read().splitlines()
    if not data: 
        return
    N, M = map(int, data[0].split())
    grid = data[1:]
    
    # Directions: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # Use 0-indexing; starting position (2,2) becomes (1,1)
    start = (1,1)
    
    # BFS for the resting positions
    queue = deque()
    queue.append(start)
    
    visited_rest = set()
    visited_rest.add(start)
    
    touched = set()
    touched.add(start)  # starting cell is touched
    
    while queue:
        i, j = queue.popleft()
        # For each of the 4 moves:
        for di, dj in directions:
            cur_i, cur_j = i, j
            # simulate sliding in this direction:
            # we will also mark all cells that are passed.
            # It is important that we stop once next is rock.
            while True:
                nxt_i = cur_i + di
                nxt_j = cur_j + dj
                
                # if next cell is rock, then stop
                if grid[nxt_i][nxt_j] == '#':
                    break
                # else if ice, then move into that cell
                cur_i, cur_j = nxt_i, nxt_j
                # mark that cell as touched
                touched.add((cur_i, cur_j))
            # The final resting square is (cur_i, cur_j). 
            if (cur_i, cur_j) not in visited_rest:
                visited_rest.add((cur_i, cur_j))
                queue.append((cur_i, cur_j))
    
    # Print count of touched ice squares.
    print(len(touched))
    
if __name__ == '__main__':
    main()