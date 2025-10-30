import sys
import bisect

# Helper functions for Segment Tree (0-based node indexing)
def build_st(tree, v, tl, tr):
    """Builds segment tree on indices [tl, tr]"""
    if tl == tr:
        tree[v] = 1 # Represents one unvisited house at this index
    else:
        tm = (tl + tr) // 2
        build_st(tree, 2*v + 1, tl, tm)
        build_st(tree, 2*v + 2, tm + 1, tr)
        tree[v] = tree[2*v + 1] + tree[2*v + 2]

def update_st(tree, v, tl, tr, pos):
    """Sets value at index pos to 0 and updates ancestors"""
    if tl == tr:
        tree[v] = 0
    else:
        tm = (tl + tr) // 2
        if pos <= tm:
            update_st(tree, 2*v + 1, tl, tm, pos)
        else:
            update_st(tree, 2*v + 2, tm + 1, tr, pos)
        tree[v] = tree[2*v + 1] + tree[2*v + 2]

def query_st_recursive(tree, v, tl, tr, query_l, query_r):
    """Finds indices in [query_l, query_r] within [tl, tr] with value 1 using recursion."""
    # Calculate the actual intersection range of the current node and the query range
    actual_query_l = max(query_l, tl)
    actual_query_r = min(query_r, tr)

    # Case 1: Actual query range is invalid or no unvisited houses in node's range
    if actual_query_l > actual_query_r or tree[v] == 0:
        return []
    
    # Case 2: Node is a leaf within the actual query range and unvisited
    if tl == tr: # Since tree[v] > 0 and it's a leaf, tree[v] must be 1
        return [tl]

    # Case 3: Node range [tl, tr] partially or fully overlaps query range [query_l, query_r]
    tm = (tl + tr) // 2
    
    # Recurse on children using the original query range [query_l, query_r]
    # The recursion's base case and calculation of actual_query_l/r handle the intersection logic
    left_indices = query_st_recursive(tree, 2*v + 1, tl, tm, query_l, query_r)
    right_indices = query_st_recursive(tree, 2*v + 2, tm + 1, tr, query_l, query_r)

    return left_indices + right_indices


# Function to get just the coordinates from a sorted list of (coord, index) tuples
def get_coords_list(sorted_list):
    return [item[0] for item in sorted_list]

# Main logic
def solve():
    # Use faster input reading
    input = sys.stdin.readline

    N, M, sx, sy = map(int, input().split())

    houses = []
    for i in range(N):
        x, y = map(int, input().split())
        houses.append((x, y, i)) # Store (x, y, original_index)

    # Group houses by Y and sort by X, build segment trees
    houses_by_y = {}
    for x, y, i in houses:
        if y not in houses_by_y:
            houses_by_y[y] = []
        houses_by_y[y].append((x, i))

    st_data_y = {} # Stores (sorted_list_by_X, segment_tree_array, coords_list_by_X)
    for y in houses_by_y:
        houses_by_y[y].sort() # Sort by X coordinate
        K = len(houses_by_y[y])
        coords_list = get_coords_list(houses_by_y[y])
        st_size = 4 * K + 5 # Safe upper bound for 0-based indexing
        tree = [0] * st_size
        # Build ST for indices 0 to K-1 of the sorted list
        build_st(tree, 0, 0, K - 1)
        st_data_y[y] = (houses_by_y[y], tree, coords_list) 

    # Group houses by X and sort by Y, build segment trees
    houses_by_x = {}
    for x, y, i in houses:
        if x not in houses_by_x:
            houses_by_x[x] = []
        houses_by_x[x].append((y, i))

    st_data_x = {} # Stores (sorted_list_by_Y, segment_tree_array, coords_list_by_Y)
    for x in houses_by_x:
        houses_by_x[x].sort() # Sort by Y coordinate
        K = len(houses_by_x[x])
        coords_list = get_coords_list(houses_by_x[x])
        st_size = 4 * K + 5 # Safe upper bound for 0-based indexing
        tree = [0] * st_size
        # Build ST for indices 0 to K-1 of the sorted list
        build_st(tree, 0, 0, K - 1)
        st_data_x[x] = (houses_by_x[x], tree, coords_list)

    current_x, current_y = sx, sy
    visited_houses = set()

    for _ in range(M):
        direction, count_str = input().split()
        count = int(count_str)

        # Determine the start and end coordinates for the current move
        start_x, start_y = current_x, current_y
        
        if direction == 'U':
            end_x, end_y = current_x, current_y + count
            # Vertical move on line x = start_x. Houses on this line are sorted by Y.
            constant_coord = start_x
            if constant_coord in st_data_x:
                sorted_list, tree, coords = st_data_x[constant_coord] # List sorted by Y, Y-coordinates
                K = len(coords) # Number of houses on this line

                # Determine the range of the changing coordinate (Y) covered by the move
                coord_min, coord_max = min(start_y, end_y), max(start_y, end_y)

                # Find the range of indices in the sorted list where the coordinate (Y) is in [coord_min, coord_max]
                idx_low = bisect.bisect_left(coords, coord_min)
                idx_high = bisect.bisect_right(coords, coord_max)

                # The relevant indices in the sorted list are from idx_low up to (but not including) idx_high.
                # The query range for the segment tree is [idx_low, idx_high - 1].
                if idx_low < idx_high:
                    # Query segment tree for unvisited house indices within the range [idx_low, idx_high-1]
                    # Pass the full range of the ST (0 to K-1) and the query range [idx_low, idx_high-1]
                    unvisited_indices_in_list = query_st_recursive(tree, 0, 0, K - 1, idx_low, idx_high - 1)

                    # Process newly found unvisited houses
                    for list_idx in unvisited_indices_in_list:
                        house_idx = sorted_list[list_idx][1]
                        # Add to global set. Set add is idempotent.
                        visited_houses.add(house_idx)
                        # Mark as visited in the segment tree at the specific index `list_idx`
                        update_st(tree, 0, 0, K - 1, list_idx)

        elif direction == 'D':
            end_x, end_y = current_x, current_y - count
            # Vertical move on line x = start_x. Houses on this line are sorted by Y.
            constant_coord = start_x
            if constant_coord in st_data_x:
                sorted_list, tree, coords = st_data_x[constant_coord] # List sorted by Y, Y-coordinates
                K = len(coords)

                coord_min, coord_max = min(start_y, end_y), max(start_y, end_y)

                idx_low = bisect.bisect_left(coords, coord_min)
                idx_high = bisect.bisect_right(coords, coord_max)

                if idx_low < idx_high:
                     unvisited_indices_in_list = query_st_recursive(tree, 0, 0, K - 1, idx_low, idx_high - 1)

                     for list_idx in unvisited_indices_in_list:
                        house_idx = sorted_list[list_idx][1]
                        visited_houses.add(house_idx)
                        update_st(tree, 0, 0, K - 1, list_idx)

        elif direction == 'L':
            end_x, end_y = current_x - count, current_y
             # Horizontal move on line y = start_y. Houses on this line are sorted by X.
            constant_coord = start_y
            if constant_coord in st_data_y:
                sorted_list, tree, coords = st_data_y[constant_coord] # List sorted by X, X-coordinates
                K = len(coords)

                coord_min, coord_max = min(start_x, end_x), max(start_x, end_x)

                idx_low = bisect.bisect_left(coords, coord_min)
                idx_high = bisect.bisect_right(coords, coord_max)

                if idx_low < idx_high:
                    unvisited_indices_in_list = query_st_recursive(tree, 0, 0, K - 1, idx_low, idx_high - 1)

                    for list_idx in unvisited_indices_in_list:
                        house_idx = sorted_list[list_idx][1]
                        visited_houses.add(house_idx)
                        update_st(tree, 0, 0, K - 1, list_idx)

        elif direction == 'R':
            end_x, end_y = current_x + count, current_y
            # Horizontal move on line y = start_y. Houses on this line are sorted by X.
            constant_coord = start_y
            if constant_coord in st_data_y:
                sorted_list, tree, coords = st_data_y[constant_coord] # List sorted by X, X-coordinates
                K = len(coords)

                coord_min, coord_max = min(start_x, end_x), max(start_x, end_x)

                idx_low = bisect.bisect_left(coords, coord_min)
                idx_high = bisect.bisect_right(coords, coord_max)

                if idx_low < idx_high:
                    unvisited_indices_in_list = query_st_recursive(tree, 0, 0, K - 1, idx_low, idx_high - 1)

                    for list_idx in unvisited_indices_in_list:
                        house_idx = sorted_list[list_idx][1]
                        visited_houses.add(house_idx)
                        update_st(tree, 0, 0, K - 1, list_idx)

        # Update Santa's current position for the next move
        current_x, current_y = end_x, end_y

    print(current_x, current_y, len(visited_houses))

if __name__ == "__main__":
    solve()