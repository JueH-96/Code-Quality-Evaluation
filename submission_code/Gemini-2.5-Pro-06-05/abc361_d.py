import sys
import collections

def solve():
    """
    This function reads the input, solves the problem using BFS, and prints the result.
    """
    # Read input from stdin
    try:
        input = sys.stdin.readline
    except (IOError, IndexError):
        pass # Fallback for non-standard execution environments

    N = int(input())
    S = input().strip()
    T = input().strip()

    # A necessary condition is that S and T must have the same number of stones of each color.
    if sorted(S) != sorted(T):
        print(-1)
        return

    # The state is defined by the full layout of N stones and 2 empty cells.
    # We represent this as a string of length N+2.
    start_layout = S + ".."
    target_layout = T + ".."

    if start_layout == target_layout:
        print(0)
        return

    # We use a queue for Breadth-First Search (BFS).
    # Each element is a tuple: (current_layout, distance_from_start).
    queue = collections.deque([(start_layout, 0)])
    
    # A set to store visited layouts to avoid cycles and redundant work.
    visited = {start_layout}

    while queue:
        current_layout, dist = queue.popleft()

        # Find the 0-indexed position of the two empty cells.
        k = current_layout.find("..")
        
        # An operation involves moving a pair of adjacent stones.
        # We iterate through all possible adjacent pairs in the N+2 grid.
        for i in range(N + 1):
            # Check if cells i and i+1 both contain stones.
            if current_layout[i] != '.' and current_layout[i+1] != '.':
                
                # To create the next state, it's efficient to convert the string to a list.
                next_layout_list = list(current_layout)
                
                # The pair of stones to be moved.
                stone1 = next_layout_list[i]
                stone2 = next_layout_list[i+1]
                
                # The original positions of the stones become empty.
                next_layout_list[i] = '.'
                next_layout_list[i+1] = '.'
                
                # The original empty cells are filled with the moved stones.
                next_layout_list[k] = stone1
                next_layout_list[k+1] = stone2
                
                next_layout = "".join(next_layout_list)

                # If we've reached the target, we're done.
                if next_layout == target_layout:
                    print(dist + 1)
                    return

                # If this is a new, unvisited state, add it to the queue.
                if next_layout not in visited:
                    visited.add(next_layout)
                    queue.append((next_layout, dist + 1))
    
    # If the queue is empty and the target was not reached, it's impossible.
    print(-1)

solve()