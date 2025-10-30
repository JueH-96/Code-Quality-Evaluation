# YOUR CODE HERE
import sys
import bisect
from collections import defaultdict

# Increase recursion depth for deep segment tree traversals, although likely not needed based on analysis
# sys.setrecursionlimit(4000) 

# Segment Tree implementation
class SegmentTree:
    """Segment tree supporting point updates and range queries for unvisited elements count.
    Each leaf node represents a house. The value 1 means unvisited, 0 means visited.
    Internal nodes store the sum of values in their range, representing the count of unvisited houses.
    """
    def __init__(self, size):
        self.N = size
        # Calculate the required size for the tree array. Using 4*N is safe.
        # Size is based on number of leaves (houses on a specific line).
        self.tree = [0] * (4 * size)
        if size > 0: # Build tree only if size > 0
            # Builds the tree initializing leaves to 1 (unvisited) and internal nodes summing children
            self._build(1, 0, size - 1)

    def _build(self, node, start, end):
        """Recursively build the segment tree."""
        if start == end:
            # Leaf node represents a single element. Initialize count to 1 (unvisited).
            self.tree[node] = 1
        else:
            mid = (start + end) // 2
            # Recursively build left and right children
            self._build(2 * node, start, mid)
            self._build(2 * node + 1, mid + 1, end)
            # Internal node stores the sum of children counts.
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]

    def update(self, idx, val):
        """Update the value of the element at index `idx` to `val`.
        Typically `val` will be 0 to mark a house as visited."""
        if self.N == 0: return # Do nothing if tree size is 0
        self._update_recursive(1, 0, self.N - 1, idx, val)

    def _update_recursive(self, node, start, end, idx, val):
        """Recursive helper for point update."""
        if start == end:
            # Leaf node found, update its value
            self.tree[node] = val
        else:
            mid = (start + end) // 2
            if start <= idx <= mid:
                # Index `idx` is in the left child's range
                self._update_recursive(2 * node, start, mid, idx, val)
            else:
                # Index `idx` is in the right child's range
                self._update_recursive(2 * node + 1, mid + 1, end, idx, val)
            # Update current node's value based on children's updated values
            self.tree[node] = self.tree[2 * node] + self.tree[2 * node + 1]
    
    def find_unvisited_indices(self, L, R):
        """Find all leaf indices corresponding to unvisited elements (value > 0) in the index range [L, R]."""
        if self.N == 0 or L > R: return [] # Return empty list if tree is empty or range is invalid
        indices = []
        # Starts the recursive search from the root (node 1) covering the full range [0, N-1]
        self._find_unvisited_recursive(1, 0, self.N - 1, L, R, indices)
        return indices

    def _find_unvisited_recursive(self, node, start, end, L, R, indices):
        """Recursive helper to find unvisited indices in query range [L, R]."""
        # Pruning: If current node's range [start, end] is completely outside query range [L, R], stop.
        # Also prune if the current node's count is 0, meaning no unvisited houses in this subtree.
        if start > R or end < L or self.tree[node] == 0:
            return

        # If it's a leaf node
        if start == end:
             # Since self.tree[node] > 0, this leaf corresponds to an unvisited house.
             # The leaf index 'start' represents the index in the original sorted list.
             # We add this index to our result list.
             indices.append(start)
        else:
            # Internal node, need to check children
            mid = (start + end) // 2
            # Recurse on left child
            self._find_unvisited_recursive(2 * node, start, mid, L, R, indices)
            # Recurse on right child
            self._find_unvisited_recursive(2 * node + 1, mid + 1, end, L, R, indices)

def solve():
    N, M, Sx, Sy = map(int, sys.stdin.readline().split())
    
    # Store coordinate info indexed by house ID for quick lookup later
    house_coords = {} 
    for i in range(N):
        X, Y = map(int, sys.stdin.readline().split())
        house_coords[i] = {'x': X, 'y': Y}

    moves = []
    for _ in range(M):
        line = sys.stdin.readline().split()
        moves.append({'dir': line[0], 'dist': int(line[1])})

    # Group houses by X and Y coordinates using defaultdict for convenience
    houses_by_x = defaultdict(list)
    houses_by_y = defaultdict(list)

    # Populate the dictionaries with house info (coordinate and ID)
    for i in range(N):
        coords = house_coords[i]
        x, y = coords['x'], coords['y']
        houses_by_x[x].append({'y': y, 'id': i})
        houses_by_y[y].append({'x': x, 'id': i})

    # Store the list index of each house within its sorted coordinate group
    # Needed for quickly finding which leaf in the Segment Tree corresponds to a house
    pos_in_x_list = [-1] * N # Initialize with -1 to detect potential issues
    pos_in_y_list = [-1] * N
    
    # Store the coordinate lists separately for efficient bisect calls
    coords_for_x = defaultdict(list) 
    coords_for_y = defaultdict(list)

    # Sort houses within each group and compute positions/coordinate lists
    for x in houses_by_x:
        # Sort houses based on Y coordinate for fixed X
        houses_by_x[x].sort(key=lambda p: p['y'])
        # Store sorted Y coordinates for binary search
        coords_for_x[x] = [p['y'] for p in houses_by_x[x]]
        # Record the index (position) of each house in this sorted list
        for idx, p in enumerate(houses_by_x[x]):
             pos_in_x_list[p['id']] = idx

    for y in houses_by_y:
        # Sort houses based on X coordinate for fixed Y
        houses_by_y[y].sort(key=lambda p: p['x'])
        # Store sorted X coordinates for binary search
        coords_for_y[y] = [p['x'] for p in houses_by_y[y]]
        # Record the index (position) of each house in this sorted list
        for idx, p in enumerate(houses_by_y[y]):
             pos_in_y_list[p['id']] = idx

    # Initialize Segment Trees for each coordinate line that contains houses
    seg_trees_x = {}
    for x in houses_by_x:
        seg_trees_x[x] = SegmentTree(len(houses_by_x[x]))

    seg_trees_y = {}
    for y in houses_by_y:
        seg_trees_y[y] = SegmentTree(len(houses_by_y[y]))

    # Keep track of visited houses using a boolean array for O(1) check
    is_visited = [False] * N
    visited_count = 0
    
    # Simulate Santa's movements starting from (Sx, Sy)
    curr_x, curr_y = Sx, Sy

    for move in moves:
        direction = move['dir']
        distance = move['dist']
        
        start_x, start_y = curr_x, curr_y
        
        # Handle each movement direction
        if direction == 'U':
            curr_y += distance
            # Check houses on the vertical line segment X = start_x, Y in [start_y, curr_y]
            if start_x in houses_by_x: # Only proceed if there are houses on this X coordinate
                target_list = houses_by_x[start_x] # List of houses on this X line
                y_coords = coords_for_x[start_x] # Precomputed list of Y coordinates for bisect
                tree = seg_trees_x[start_x] # The segment tree for this X line
                
                y_min, y_max = start_y, curr_y # Y range of the movement segment
                
                # Find the range of list indices corresponding to the Y range using binary search
                idx_start = bisect.bisect_left(y_coords, y_min)
                idx_end = bisect.bisect_right(y_coords, y_max)
                
                # Query segment tree for unvisited house indices within the found index range [idx_start, idx_end - 1]
                unvisited_leaf_indices = tree.find_unvisited_indices(idx_start, idx_end - 1)
                
                # Process each newly found unvisited house
                for leaf_idx in unvisited_leaf_indices:
                    house_id = target_list[leaf_idx]['id']
                    # Double check using is_visited array, although Segment Tree should only return unvisited ones.
                    # This check ensures that if somehow a visited house leaf index is returned (e.g., bug), we don't process it again.
                    if not is_visited[house_id]:
                        is_visited[house_id] = True # Mark as visited
                        visited_count += 1 # Increment count
                        
                        # Mark house as visited in the current segment tree (set leaf value to 0)
                        tree.update(leaf_idx, 0)
                        
                        # Mark house as visited in the corresponding segment tree for the other dimension (Y)
                        house_info = house_coords[house_id]
                        other_y = house_info['y']
                        if other_y in seg_trees_y: # Check if a tree exists for this Y coord
                             other_tree = seg_trees_y[other_y]
                             other_leaf_idx = pos_in_y_list[house_id]
                             # Check if index is valid (-1 indicates error or not found) before updating
                             if other_leaf_idx != -1:
                                other_tree.update(other_leaf_idx, 0)

        elif direction == 'D':
            curr_y -= distance
            # Check houses on the vertical line segment X = start_x, Y in [curr_y, start_y]
            if start_x in houses_by_x:
                target_list = houses_by_x[start_x]
                y_coords = coords_for_x[start_x]
                tree = seg_trees_x[start_x]

                y_min, y_max = curr_y, start_y # Y range for Down movement

                idx_start = bisect.bisect_left(y_coords, y_min)
                idx_end = bisect.bisect_right(y_coords, y_max)

                unvisited_leaf_indices = tree.find_unvisited_indices(idx_start, idx_end - 1)
                
                for leaf_idx in unvisited_leaf_indices:
                    house_id = target_list[leaf_idx]['id']
                    if not is_visited[house_id]:
                        is_visited[house_id] = True
                        visited_count += 1
                        tree.update(leaf_idx, 0)
                        
                        house_info = house_coords[house_id]
                        other_y = house_info['y']
                        if other_y in seg_trees_y:
                             other_tree = seg_trees_y[other_y]
                             other_leaf_idx = pos_in_y_list[house_id]
                             if other_leaf_idx != -1:
                                other_tree.update(other_leaf_idx, 0)

        elif direction == 'L':
            curr_x -= distance
            # Check houses on the horizontal line segment Y = start_y, X in [curr_x, start_x]
            if start_y in houses_by_y:
                target_list = houses_by_y[start_y]
                x_coords = coords_for_y[start_y]
                tree = seg_trees_y[start_y]

                x_min, x_max = curr_x, start_x # X range for Left movement

                idx_start = bisect.bisect_left(x_coords, x_min)
                idx_end = bisect.bisect_right(x_coords, x_max)

                unvisited_leaf_indices = tree.find_unvisited_indices(idx_start, idx_end - 1)
                
                for leaf_idx in unvisited_leaf_indices:
                    house_id = target_list[leaf_idx]['id']
                    if not is_visited[house_id]:
                        is_visited[house_id] = True
                        visited_count += 1
                        tree.update(leaf_idx, 0)

                        house_info = house_coords[house_id]
                        other_x = house_info['x']
                        if other_x in seg_trees_x:
                             other_tree = seg_trees_x[other_x]
                             other_leaf_idx = pos_in_x_list[house_id]
                             if other_leaf_idx != -1:
                                other_tree.update(other_leaf_idx, 0)


        elif direction == 'R':
            curr_x += distance
             # Check houses on the horizontal line segment Y = start_y, X in [start_x, curr_x]
            if start_y in houses_by_y:
                target_list = houses_by_y[start_y]
                x_coords = coords_for_y[start_y]
                tree = seg_trees_y[start_y]

                x_min, x_max = start_x, curr_x # X range for Right movement
                
                idx_start = bisect.bisect_left(x_coords, x_min)
                idx_end = bisect.bisect_right(x_coords, x_max)

                unvisited_leaf_indices = tree.find_unvisited_indices(idx_start, idx_end - 1)
                
                for leaf_idx in unvisited_leaf_indices:
                    house_id = target_list[leaf_idx]['id']
                    if not is_visited[house_id]:
                        is_visited[house_id] = True
                        visited_count += 1
                        tree.update(leaf_idx, 0)
                        
                        house_info = house_coords[house_id]
                        other_x = house_info['x']
                        if other_x in seg_trees_x:
                             other_tree = seg_trees_x[other_x]
                             other_leaf_idx = pos_in_x_list[house_id]
                             if other_leaf_idx != -1:
                                other_tree.update(other_leaf_idx, 0)

    # Output final position and count of distinct visited houses
    print(f"{curr_x} {curr_y} {visited_count}")

solve()