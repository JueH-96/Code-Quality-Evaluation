import collections

class FenwickTree:
    def __init__(self, size: int):
        # size is the number of elements (e.g., M unique y-coords)
        # The tree is 1-indexed internally, so its array length is size + 1
        self.tree = [0] * (size + 1)
        self.size = size

    def update(self, idx: int, val: int):
        # idx is 0-based index of the element to update
        # This method ensures that tree[idx] (conceptually) holds at least val,
        # and propagates this maximum upwards.
        
        # Basic bounds check, though careful usage should prevent out-of-bounds.
        if not (0 <= idx < self.size):
            return

        idx += 1 # Convert to 1-based index for Fenwick tree array
        while idx <= self.size: # Iterate through tree nodes this index contributes to
            self.tree[idx] = max(self.tree[idx], val)
            idx += idx & (-idx) # Move to the next relevant node

    def query(self, idx: int) -> int:
        # idx is 0-based index, query returns max in [0, idx] (inclusive)
        
        # If idx is negative (e.g. compressed_y - 1 when compressed_y is 0),
        # it means an empty valid range, so max is 0.
        if idx < 0:
            return 0
        
        # Caller should ensure idx < self.size. If idx is too large, cap it.
        # However, standard Fenwick tree query doesn't cap, relies on correct input.
        # For this problem, max query idx is M-1 (or M-2 after subtraction).
        # idx = min(idx, self.size - 1) # Optional: defensive capping

        idx += 1 # Convert to 1-based index
        res = 0
        while idx > 0:
            res = max(res, self.tree[idx])
            idx -= idx & (-idx) # Move to the "parent" or node summarizing a larger prefix
        return res

class Solution:
  def maxPathLength(self, coordinates: list[list[int]], k: int) -> int:
    n = len(coordinates)

    if n == 0: # Should not happen based on constraints (1 <= n)
        return 0
    
    # Coordinate compression for y-coordinates
    # Collect all unique y-coordinates, sort them, and map to 0..M-1
    unique_y_coords = sorted(list(set(coord[1] for coord in coordinates)))
    y_map = {y_coord: i for i, y_coord in enumerate(unique_y_coords)}
    M = len(unique_y_coords) # Number of unique y-coordinates

    # Calculate Lin: length of longest increasing path ending at point i
    Lin = [0] * n
    # Augment points with original indices: (x, y, original_index)
    points_asc = []
    for i in range(n):
        points_asc.append((coordinates[i][0], coordinates[i][1], i))
    
    # Sort by x ascending, then y ascending
    points_asc.sort()

    ft_in = FenwickTree(M)
    
    # Process points in batches of same x-coordinate
    i = 0
    while i < n:
        j = i
        # Find end of current batch (all points with points_asc[i][0] as x-coordinate)
        while j < n and points_asc[j][0] == points_asc[i][0]:
            j += 1
        
        # Current batch is points_asc[i...j-1]
        # Phase 1: Queries for all points in the batch
        # Store (compressed_y, calculated_Lin_value) for updates in Phase 2
        batch_updates_info = [] 
        for batch_idx in range(i, j):
            _px, py, original_idx = points_asc[batch_idx] # _px is current x
            compressed_y = y_map[py]
            
            # Query for max Lin of points (x_prev, y_prev) where x_prev < _px and y_prev < py
            # This means compressed_y_prev < compressed_y.
            # Query max in Fenwick tree for indices [0, compressed_y - 1]
            max_prev_L = ft_in.query(compressed_y - 1)
            
            Lin[original_idx] = 1 + max_prev_L
            batch_updates_info.append((compressed_y, Lin[original_idx]))
        
        # Phase 2: Updates for all points in the batch
        for comp_y_to_update, val_to_update in batch_updates_info:
            ft_in.update(comp_y_to_update, val_to_update)
        
        i = j # Move to the start of the next batch

    # Calculate Lout: length of longest increasing path starting at point i
    Lout = [0] * n
    points_desc = [] # New list for descending sort
    for idx_orig in range(n):
        points_desc.append((coordinates[idx_orig][0], coordinates[idx_orig][1], idx_orig))

    # Sort by x descending, then y descending
    points_desc.sort(key=lambda p: (p[0], p[1]), reverse=True)
    
    ft_out = FenwickTree(M)

    # Process points in batches of same x-coordinate (now in descending x order)
    i = 0
    while i < n:
        j = i
        while j < n and points_desc[j][0] == points_desc[i][0]:
            j += 1
        
        batch_updates_info = []
        # Phase 1: Queries
        for batch_idx in range(i, j):
            _px, py, original_idx = points_desc[batch_idx] # _px is current x
            compressed_y = y_map[py]

            # For Lout, we need points (x_next, y_next) where x_next > _px and y_next > py.
            # To use prefix max query Fenwick tree for y_next > py:
            # Map y-coordinates: transformed_y = M - 1 - compressed_y.
            # Condition y_next > py becomes transformed_y_next < transformed_y_current.
            # Query max in Fenwick tree for indices [0, (M - 1 - compressed_y) - 1]
            current_transformed_y = M - 1 - compressed_y
            max_next_L = ft_out.query(current_transformed_y - 1)
            
            Lout[original_idx] = 1 + max_next_L
            
            # Store info for update phase, using the transformed y index
            batch_updates_info.append((current_transformed_y, Lout[original_idx]))

        # Phase 2: Updates
        for transf_comp_y_to_update, val_to_update in batch_updates_info:
            ft_out.update(transf_comp_y_to_update, val_to_update)
        
        i = j
            
    # The result is Lin[k] + Lout[k] - 1, as point P_k is counted in both paths.
    return Lin[k] + Lout[k] - 1