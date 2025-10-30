import sys

# Fenwick Tree (Binary Indexed Tree) implementation
class FenwickTree:
    def __init__(self, size):
        # The tree array is 1-indexed, so size + 1
        self.tree = [0] * (size + 1)
        self.size = size

    def update(self, index, delta):
        # Convert 0-based index to 1-based for BIT operations
        index += 1
        while index <= self.size:
            self.tree[index] += delta
            index += index & (-index)

    def query(self, index):
        # Convert 0-based index to 1-based for BIT operations
        # query(index) returns sum of elements from 0 to index (inclusive)
        index += 1
        s = 0
        while index > 0:
            s += self.tree[index]
            index -= index & (-index)
        return s

def solve():
    N = int(sys.stdin.readline())
    intervals = []
    for _ in range(N):
        l, r = map(int, sys.stdin.readline().split())
        intervals.append((l, r))

    # Coordinate compression
    # Collect all unique l and r coordinates
    all_coords = set()
    for l, r in intervals:
        all_coords.add(l)
        all_coords.add(r)
    
    # Sort and map to 0-based indices
    sorted_coords = sorted(list(all_coords))
    coord_to_idx = {val: i for i, val in enumerate(sorted_coords)}
    M = len(sorted_coords) # Size of the compressed coordinate space

    # Create events for the sweep line algorithm
    # Event format: (coordinate, type, original_r_value)
    # type 0 for start (l_i), type 1 for end (r_i)
    # Sorting order: process start events before end events at the same coordinate
    events = []
    for l, r in intervals:
        events.append((l, 0, r)) # (coordinate, event_type, original_r_value)
        events.append((r, 1, r)) # (coordinate, event_type, original_r_value)
    
    # Sort events.
    # Python's default tuple sorting ensures:
    # 1. Primary sort key: coordinate (event_[0])
    # 2. Secondary sort key: event_type (event_[1]), so '0' (start) comes before '1' (end)
    events.sort()

    total_intersections = 0
    # Initialize Fenwick Tree, operating on compressed indices 0 to M-1
    bit = FenwickTree(M) 

    for coord, event_type, original_r_value in events:
        if event_type == 0:  # Start of an interval [coord, original_r_value] (coord is l_i)
            # Query for active intervals [L, R] such that R >= current 'coord' (l_i).
            # The Fenwick tree stores counts of r_values (as compressed indices) of active intervals.
            # We need to sum counts for all r_values whose compressed index is >= compressed index of l_i.
            
            # Get the compressed index of the current l_i (which is 'coord')
            l_idx_in_bit = coord_to_idx[coord]
            
            # The count of active intervals whose r-value is >= l_i is:
            # (total count of all active intervals) - (count of active intervals with r-value < l_i)
            # bit.query(M-1) gives the sum up to the maximum possible compressed index (total active intervals).
            # bit.query(l_idx_in_bit - 1) gives the sum up to (l_idx_in_bit - 1),
            # effectively counting active intervals whose r-value is strictly less than l_i.
            total_intersections += bit.query(M - 1) - bit.query(l_idx_in_bit - 1)
            
            # Add the current interval's r_value to the Fenwick Tree.
            # Use its compressed index.
            r_idx_in_bit = coord_to_idx[original_r_value]
            bit.update(r_idx_in_bit, 1)

        else:  # End of an interval (coord is r_i)
            # Remove this interval's r_value from the Fenwick Tree.
            # This interval is no longer active.
            r_idx_in_bit = coord_to_idx[original_r_value]
            bit.update(r_idx_in_bit, -1)

    sys.stdout.write(str(total_intersections) + "
")

solve()