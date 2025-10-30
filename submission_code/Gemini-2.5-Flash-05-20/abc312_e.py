import sys

# For fast I/O
input = sys.stdin.readline

MAX_COORD = 100 # Coordinates are 0 to 100 inclusive.

class SegmentTreeNode:
    def __init__(self):
        self.values = set()  # Stores indices of cuboids whose intervals fully cover this node's range.
        self.left_child = None
        self.right_child = None

class SegmentTree:
    def __init__(self, coord_range_max):
        # Tree covers [0, coord_range_max) interval (e.g., [0, 101) to cover 0-100 coordinates)
        self.root = SegmentTreeNode()
        self.coord_range_max = coord_range_max
        self._build(self.root, 0, self.coord_range_max)

    def _build(self, node, low, high):
        if low + 1 == high: # Leaf node, represents a single unit interval [low, high)
            return
        mid = (low + high) // 2
        node.left_child = SegmentTreeNode()
        node.right_child = SegmentTreeNode()
        self._build(node.left_child, low, mid)
        self._build(node.right_child, mid, high)

    def add_interval(self, x1, x2, val):
        # Add interval [x1, x2)
        if x1 >= x2: return # Empty interval
        self._add_interval_recursive(self.root, 0, self.coord_range_max, x1, x2, val)

    def _add_interval_recursive(self, node, node_low, node_high, x1, x2, val):
        # Current node's range: [node_low, node_high)
        # Interval to add: [x1, x2)
        if x1 <= node_low and node_high <= x2: # Node's range is fully contained within the interval
            node.values.add(val)
            return

        if node_high <= x1 or x2 <= node_low: # No overlap
            return

        # Partial overlap, recurse
        mid = (node_low + node_high) // 2
        self._add_interval_recursive(node.left_child, node_low, mid, x1, x2, val)
        self._add_interval_recursive(node.right_child, mid, node_high, x1, x2, val)

    def remove_interval(self, x1, x2, val):
        # Remove interval [x1, x2)
        if x1 >= x2: return # Empty interval
        self._remove_interval_recursive(self.root, 0, self.coord_range_max, x1, x2, val)

    def _remove_interval_recursive(self, node, node_low, node_high, x1, x2, val):
        # Current node's range: [node_low, node_high)
        # Interval to remove: [x1, x2)
        if x1 <= node_low and node_high <= x2: # Node's range is fully contained within the interval
            node.values.remove(val)
            return

        if node_high <= x1 or x2 <= node_low: # No overlap
            return

        # Partial overlap, recurse
        mid = (node_low + node_high) // 2
        self._remove_interval_recursive(node.left_child, node_low, mid, x1, x2, val)
        self._remove_interval_recursive(node.right_child, mid, node_high, x1, x2, val)

    def query_overlaps(self, x1, x2):
        # Query interval: [x1, x2)
        if x1 >= x2: return set() # Empty interval
        result_indices = set()
        self._query_overlaps_recursive(self.root, 0, self.coord_range_max, x1, x2, result_indices)
        return result_indices

    def _query_overlaps_recursive(self, node, node_low, node_high, x1, x2, result_indices):
        # Current node's range: [node_low, node_high)
        # Query interval: [x1, x2)
        
        if node_high <= x1 or x2 <= node_low: # No overlap
            return
        
        # Add values from this node directly (these are intervals fully covering this node's range)
        result_indices.update(node.values)

        if node_low + 1 == node_high: # Leaf node, no further children
            return

        # Partial overlap or query interval contains node's range, recurse
        mid = (node_low + node_high) // 2
        self._query_overlaps_recursive(node.left_child, node_low, mid, x1, x2, result_indices)
        self._query_overlaps_recursive(node.right_child, mid, node_high, x1, x2, result_indices)


def solve():
    N = int(input())
    cuboids = []
    for _ in range(N):
        x1, y1, z1, x2, y2, z2 = map(int, input().split())
        cuboids.append((x1, y1, z1, x2, y2, z2))

    answers = [0] * N

    # Group cuboids by their faces' coordinates
    # Each list entry will be (dim1_min, dim1_max, dim2_min, dim2_max, cuboid_idx)
    # The dimensions are for the 2D projection on the plane.

    # For X-plane adjacency (YZ-plane projections)
    # x_faces_right[x_coord]: cuboids with x2 = x_coord
    # x_faces_left[x_coord]: cuboids with x1 = x_coord
    x_faces_right = [[] for _ in range(MAX_COORD + 1)] 
    x_faces_left = [[] for _ in range(MAX_COORD + 1)]  

    # For Y-plane adjacency (XZ-plane projections)
    # y_faces_up[y_coord]: cuboids with y2 = y_coord
    # y_faces_down[y_coord]: cuboids with y1 = y_coord
    y_faces_up = [[] for _ in range(MAX_COORD + 1)]    
    y_faces_down = [[] for _ in range(MAX_COORD + 1)]  

    # For Z-plane adjacency (XY-plane projections)
    # z_faces_front[z_coord]: cuboids with z2 = z_coord
    # z_faces_back[z_coord]: cuboids with z1 = z_coord
    z_faces_front = [[] for _ in range(MAX_COORD + 1)] 
    z_faces_back = [[] for _ in range(MAX_COORD + 1)]  

    for i, (x1, y1, z1, x2, y2, z2) in enumerate(cuboids):
        # YZ-plane projection: (y_min, y_max, z_min, z_max, cuboid_idx)
        x_faces_right[x2].append((y1, y2, z1, z2, i)) 
        x_faces_left[x1].append((y1, y2, z1, z2, i))

        # XZ-plane projection: (x_min, x_max, z_min, z_max, cuboid_idx)
        y_faces_up[y2].append((x1, x2, z1, z2, i))   
        y_faces_down[y1].append((x1, x2, z1, z2, i))

        # XY-plane projection: (x_min, x_max, y_min, y_max, cuboid_idx)
        z_faces_front[z2].append((x1, x2, y1, y2, i)) 
        z_faces_back[z1].append((x1, x2, y1, y2, i))

    # Process adjacency for each axis
    for axis in range(3): # 0:X, 1:Y, 2:Z
        for coord in range(MAX_COORD + 1):
            events = []
            
            set1_data = [] # Cuboids whose one face ends at this coord (e.g., x2=coord)
            set2_data = [] # Cuboids whose one face starts at this coord (e.g., x1=coord)

            if axis == 0: # X-axis adjacency (YZ-plane projections)
                set1_data = x_faces_right[coord]
                set2_data = x_faces_left[coord]
            elif axis == 1: # Y-axis adjacency (XZ-plane projections)
                set1_data = y_faces_up[coord]
                set2_data = y_faces_down[coord]
            else: # Z-axis adjacency (XY-plane projections)
                set1_data = z_faces_front[coord]
                set2_data = z_faces_back[coord]
            
            # Events for the 2D sweep-line
            # event format: (primary_dim_val, type, secondary_dim_min, secondary_dim_max, cuboid_idx, set_type)
            # type: 0 for start of range, 1 for end of range
            # set_type: 0 for set1 (ending faces), 1 for set2 (starting faces)

            for d1_min, d1_max, d2_min, d2_max, idx in set1_data:
                events.append((d1_min, 0, d2_min, d2_max, idx, 0)) # Start of rectangle from set1
                events.append((d1_max, 1, d2_min, d2_max, idx, 0)) # End of rectangle from set1
            for d1_min, d1_max, d2_min, d2_max, idx in set2_data:
                events.append((d1_min, 0, d2_min, d2_max, idx, 1)) # Start of rectangle from set2
                events.append((d1_max, 1, d2_min, d2_max, idx, 1)) # End of rectangle from set2
            
            # Sort events: primary_dim_val, then type (start before end), then secondary_dim_min, then secondary_dim_max.
            # Python's sort is stable and sorts tuples lexicographically, which works for (val, type, ...)
            events.sort()

            # Segment tree operates on the secondary dimension range [0, MAX_COORD+1)
            # e.g., for X-axis adjacency, primary is Y, secondary is Z. So tree is on Z.
            seg_tree_set1 = SegmentTree(MAX_COORD + 1) 
            seg_tree_set2 = SegmentTree(MAX_COORD + 1)
            
            for d1_val, type, d2_min, d2_max, idx, set_type in events:
                # The range for the segment tree is [d2_min, d2_max)
                # These ranges are exclusive on d_max, which is standard for segment trees on unit intervals.
                if type == 0: # Start of rectangle
                    if set_type == 0: # From set1 (ending faces)
                        # Query set2 for overlaps in secondary dimension range
                        overlapping_indices = seg_tree_set2.query_overlaps(d2_min, d2_max)
                        for other_idx in overlapping_indices:
                            answers[idx] += 1
                            answers[other_idx] += 1
                        seg_tree_set1.add_interval(d2_min, d2_max, idx)
                    else: # From set2 (starting faces)
                        # Query set1 for overlaps in secondary dimension range
                        overlapping_indices = seg_tree_set1.query_overlaps(d2_min, d2_max)
                        for other_idx in overlapping_indices:
                            answers[idx] += 1
                            answers[other_idx] += 1
                        seg_tree_set2.add_interval(d2_min, d2_max, idx)
                else: # End of rectangle
                    if set_type == 0:
                        seg_tree_set1.remove_interval(d2_min, d2_max, idx)
                    else:
                        seg_tree_set2.remove_interval(d2_min, d2_max, idx)

    for ans_val in answers:
        sys.stdout.write(str(ans_val) + "
")

solve()