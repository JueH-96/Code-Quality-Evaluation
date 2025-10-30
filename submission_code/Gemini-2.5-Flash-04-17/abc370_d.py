# YOUR CODE HERE
from sortedcontainers import SortedSet
import sys

# Function to solve the problem
def solve():
    # Read H, W, Q from the first line of input
    line = sys.stdin.readline().split()
    H = int(line[0])
    W = int(line[1])
    Q = int(line[2])

    # Initial number of walls is the total number of cells
    wall_count = H * W

    # Data structure to store wall locations efficiently.
    # We use two lists of SortedSets:
    # row_walls[r] is a SortedSet containing the column indices c where a wall exists at (r, c).
    # col_walls[c] is a SortedSet containing the row indices r where a wall exists at (r, c).
    # SortedSet from sortedcontainers allows O(log N) operations for membership test, addition, removal,
    # and finding predecessor/successor, where N is the size of the set.
    # Initially, walls exist in all cells.
    row_walls = [SortedSet(range(W)) for _ in range(H)]
    col_walls = [SortedSet(range(H)) for _ in range(W)]

    # Process Q queries
    for _ in range(Q):
        # Read the query coordinates R and C (1-based)
        line = sys.stdin.readline().split()
        R = int(line[0])
        C = int(line[1])
        
        # Convert 1-based coordinates to 0-based indices
        r_q = R - 1
        c_q = C - 1

        # Check if there is a wall at the queried cell (r_q, c_q)
        # We can check if the column index c_q is in the SortedSet for row r_q
        if c_q in row_walls[r_q]:
            # Case 1: A wall exists at (r_q, c_q)
            # Destroy this wall by removing its coordinates from both the row and column sets
            row_walls[r_q].remove(c_q) # Remove column c_q from the set of walls in row r_q
            col_walls[c_q].remove(r_q) # Remove row r_q from the set of walls in column c_q
            # Decrement the total count of walls
            wall_count -= 1
        else:
            # Case 2: No wall exists at (r_q, c_q). The cell is empty.
            # Destroy the first walls encountered when looking up, down, left, and right.
            # These four searches occur based on the grid state *before* any walls are removed in this step.
            
            # Look Up: Find the wall (r_up, c_q) with the largest row index r_up strictly less than r_q.
            # This is the predecessor of r_q in the set of row indices in col_walls[c_q].
            # bisect_left(value) returns the index where value would be inserted to maintain order.
            # Elements at indices less than this are strictly less than value.
            idx_up = col_walls[c_q].bisect_left(r_q)
            # If idx_up > 0, it means there is at least one element (row index) in col_walls[c_q] that is less than r_q.
            # The element at index idx_up - 1 is the largest such element (the predecessor).
            if idx_up > 0:
                r_up = col_walls[c_q][idx_up - 1]
                # Destroy the wall at (r_up, c_q)
                col_walls[c_q].remove(r_up) # Remove row index r_up from the set of walls in column c_q
                row_walls[r_up].remove(c_q) # Remove column index c_q from the set of walls in row r_up
                wall_count -= 1 # Decrement total wall count

            # Look Down: Find the wall (r_down, c_q) with the smallest row index r_down strictly greater than r_q.
            # This is the successor of r_q in the set of row indices in col_walls[c_q].
            # bisect_right(value) returns the index where value would be inserted such that elements before it are <= value.
            # Elements at indices greater than or equal to this are strictly greater than value.
            idx_down = col_walls[c_q].bisect_right(r_q)
            # If idx_down < len(col_walls[c_q]), it means there is at least one element (row index) in col_walls[c_q]
            # at or after this index, which must be strictly greater than r_q.
            # The element at index idx_down is the smallest such element (the successor).
            if idx_down < len(col_walls[c_q]):
                r_down = col_walls[c_q][idx_down]
                # Destroy the wall at (r_down, c_q)
                col_walls[c_q].remove(r_down)
                row_walls[r_down].remove(c_q)
                wall_count -= 1

            # Look Left: Find the wall (r_q, c_left) with the largest column index c_left strictly less than c_q.
            # This is the predecessor of c_q in the set of column indices in row_walls[r_q].
            idx_left = row_walls[r_q].bisect_left(c_q)
            if idx_left > 0: # If idx_left > 0, there is at least one element < c_q
                c_left = row_walls[r_q][idx_left - 1] # The predecessor
                # Destroy the wall at (r_q, c_left)
                row_walls[r_q].remove(c_left)
                col_walls[c_left].remove(r_q)
                wall_count -= 1

            # Look Right: Find the wall (r_q, c_right) with the smallest column index c_right strictly greater than c_q.
            # This is the successor of c_q in the set of column indices in row_walls[r_q].
            idx_right = row_walls[r_q].bisect_right(c_q)
            if idx_right < len(row_walls[r_q]): # If idx_right < len(...), there is at least one element > c_q
                c_right = row_walls[r_q][idx_right] # The successor
                # Destroy the wall at (r_q, c_right)
                row_walls[r_q].remove(c_right)
                col_walls[c_right].remove(r_q)
                wall_count -= 1

    # After processing all queries, print the final count of remaining walls
    print(wall_count)

# Call the solve function to run the program
solve()