import sys
# Setting a higher recursion depth is generally not needed for iterative solutions like this one using Fenwick trees.
# sys.setrecursionlimit(2000) 

from typing import List

# MaxFenwickTree (Binary Indexed Tree for Range Maximum Query)
class MaxFenwickTree:
    """
    A Fenwick tree (Binary Indexed Tree) implementation that supports
    point updates (setting maximum value) and prefix maximum queries.
    It uses 0-based indexing for inputs (indices i) but converts
    to 1-based indexing internally for the standard BIT algorithms.
    Assumes values (path lengths) are non-negative.
    """
    def __init__(self, size: int):
        """
        Initializes the Fenwick tree.
        :param size: The number of elements (maximum index + 1).
                     Indices managed are 0 to size-1. The tree will handle these indices.
        """
        # Internal tree array size is size+1 to accommodate 1-based indexing logic.
        self.tree = [0] * (size + 1)
        # Store the maximum 0-based index that can be managed.
        self.max_idx = size - 1
        # Store the size of the internal tree array (capacity + 1).
        self.tree_size = size + 1

    def update(self, i: int, val: int):
        """
        Updates the maximum value associated with index i. If the new value `val`
        is greater than the current maximum value implicitly represented at index `i`
        and its ancestors in the tree, it updates the tree accordingly.
        :param i: The 0-based index to update. Should be in range [0, max_idx].
        :param val: The value (a path length) to update with.
        """
        # Convert to 1-based index for internal BIT logic
        i += 1  
        # Propagate the maximum value up the tree
        while i < self.tree_size:
            self.tree[i] = max(self.tree[i], val)
            # Move to the next node in the update path (node responsible for a larger range including i)
            i += i & (-i) # Add the least significant bit

    def query(self, i: int) -> int:
        """
        Queries the maximum value in the prefix range [0, i] (inclusive).
        :param i: The 0-based index defining the end of the prefix range.
        :return: The maximum value found in the range [0, i]. Returns 0 if i < 0 (empty range).
        """
        if i < 0:
            # Convention: Maximum value in an empty range is 0.
            # Since path lengths are >= 1, 0 indicates no valid predecessor path found.
            return 0  

        # Ensure the query index doesn't exceed the managed range.
        i = min(i, self.max_idx)

        # Convert to 1-based index
        i += 1  
        max_val = 0
        # Traverse down the tree (in terms of indices) to aggregate the maximum
        while i > 0:
            max_val = max(max_val, self.tree[i])
            # Move to the node responsible for the prefix ending before the current node's range
            i -= i & (-i) # Subtract the least significant bit
        return max_val

class Solution:
    """
    Solves the problem of finding the maximum length of an increasing path
    in a set of 2D coordinates, with the constraint that the path must contain
    a specific point coordinates[k]. An increasing path requires both x and y 
    coordinates to strictly increase between consecutive points.
    """

    def _calculate_dp(self, n: int, aug_coords_sorted: List[tuple], y_map: dict, m: int) -> List[int]:
        """
        Helper function to calculate the lengths of longest increasing paths ending at each point.
        This function is used for both the forward pass (paths ending at points) and, with
        negated coordinates, for the backward pass (paths starting at points).

        It uses a Fenwick tree (MaxFenwickTree) for efficient O(log m) querying of the 
        maximum path length among valid predecessors.

        :param n: Number of coordinates.
        :param aug_coords_sorted: List of tuples `(primary_key, secondary_key, original_index)`, 
                                  sorted appropriately. For the forward pass, this is `(x, y, index)` 
                                  sorted by x then y. For the backward pass, it's `(-x, -y, index)` 
                                  sorted by -x then -y.
        :param y_map: Dictionary mapping the secondary key values (y or -y) to compressed 0-based indices 
                      suitable for the Fenwick tree.
        :param m: The number of unique secondary key values (size needed for the Fenwick tree).
        :return: A list `dp` of size `n`, where `dp[i]` stores the length of the longest path 
                 ending at the point `coordinates[i]` (using the original index `i`).
        """
        # Initialize the Fenwick tree of size m (number of unique y or -y values)
        max_dp_tree = MaxFenwickTree(m)
        # Initialize DP array to store results, indexed by the original point index.
        dp = [0] * n  
        
        i = 0
        while i < n:
            # Process points in batches that share the same primary sorting key (x for fwd, -x for bwd)
            # This ensures correct handling based on the strict inequality requirement (x_i < x_{i+1}).
            current_primary_key = aug_coords_sorted[i][0] 
            
            # Store updates for this batch temporarily.
            # We apply updates only after processing all points in the batch.
            batch_updates = [] # Stores tuples: (y_comp_idx, dp_value, original_idx)
            
            batch_start_index = i
            # Iterate through all points belonging to the current batch (same primary key)
            while i < n and aug_coords_sorted[i][0] == current_primary_key:
                # Unpack point info: primary_key (x or -x), secondary_key (y or -y), original_index
                primary_key_val, secondary_key_val, original_index = aug_coords_sorted[i]
                
                # Get the compressed index for the secondary key using the provided map.
                y_comp = y_map[secondary_key_val]
                
                # Query the Fenwick tree to find the maximum path length among all points processed *before* 
                # the current batch that could precede this point in an increasing path.
                # This requires primary_key_prev < current_primary_key (handled by batch processing)
                # and secondary_key_prev < secondary_key_val.
                # The query `query(y_comp - 1)` finds the max dp among points with compressed y index < y_comp.
                max_prev_dp = max_dp_tree.query(y_comp - 1)
                
                # The length of the longest path ending at the current point is 1 (for the point itself)
                # plus the maximum length of a path ending at a valid predecessor.
                current_dp = 1 + max_prev_dp
                
                # Schedule this result to be updated into the Fenwick tree and DP array later.
                batch_updates.append((y_comp, current_dp, original_index))
                
                # Move to the next point in the sorted list
                i += 1 
            
            # After processing all points in the batch, apply the collected updates.
            # This delay ensures that points within the same batch (sharing the same x or -x) 
            # do not influence each other's DP calculation, correctly enforcing the strict inequality.
            for y_comp_update, dp_value, original_idx_update in batch_updates:
                # Update the Fenwick tree with the new maximum path length found for this y_comp index.
                max_dp_tree.update(y_comp_update, dp_value)
                # Store the calculated DP value in the result array, mapped back to the original index.
                dp[original_idx_update] = dp_value
                
        return dp

    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        """
        Calculates the maximum length of an increasing path passing through the point `coordinates[k]`.
        An increasing path is defined by points (x_i, y_i) such that x_i < x_{i+1} and y_i < y_{i+1}.
        
        :param coordinates: A list of 2D points, where coordinates[i] = [x_i, y_i].
        :param k: The index of the point that must be included in the path.
        :return: The maximum length of such an increasing path.
        """
        n = len(coordinates)
        # Handle edge cases: If n is 0 or 1, the path length is simply n.
        # The constraints state N >= 1, so n=0 might not occur, but handling it is robust.
        if n <= 1:
            return n

        # --- Forward Pass: Calculate Longest Path ENDING at each point ---
        
        # 1. Augment coordinates with original index: (x, y, original_index)
        # This preserves the original index needed for the final result and dp array mapping.
        aug_coords_fwd = [(coordinates[i][0], coordinates[i][1], i) for i in range(n)]
        # 2. Sort points primarily by x-coordinate, secondarily by y-coordinate.
        # This order is crucial for the DP calculation.
        aug_coords_fwd.sort()
        
        # 3. Perform coordinate compression on y-coordinates.
        # This maps potentially large y-values to a compact range [0, m-1] for the Fenwick tree.
        all_y_fwd = sorted(list(set(c[1] for c in coordinates))) # Get unique sorted y values
        y_map_fwd = {y: i for i, y in enumerate(all_y_fwd)} # Map y value to its 0-based rank
        m_fwd = len(all_y_fwd) # Number of unique y values (size for Fenwick tree)
        
        # 4. Calculate dp_fwd[i] = length of the longest increasing path ending at coordinates[i].
        dp_fwd = self._calculate_dp(n, aug_coords_fwd, y_map_fwd, m_fwd)

        # --- Backward Pass: Calculate Longest Path STARTING at each point ---
        # This is mathematically equivalent to finding the longest path ending at each point 
        # in a reversed graph. We achieve this by:
        #   a) Negating both x and y coordinates: (x, y) -> (-x, -y).
        #   b) Running the same `_calculate_dp` logic on these negated coordinates.
        # An increasing path in (-x, -y) corresponds to a decreasing path in (x, y), 
        # which, when read in reverse, is an increasing path starting from the original point.
        
        # 1. Create augmented coordinates using negated values: (-x, -y, original_index)
        aug_coords_bwd = [(-coordinates[i][0], -coordinates[i][1], i) for i in range(n)]
        # 2. Sort primarily by -x, secondarily by -y.
        aug_coords_bwd.sort()
        
        # 3. Perform coordinate compression on the negated y-coordinates (-y).
        all_y_bwd = sorted(list(set(-c[1] for c in coordinates))) # Get unique sorted -y values
        y_map_bwd = {neg_y: i for i, neg_y in enumerate(all_y_bwd)} # Map -y value to its rank
        m_bwd = len(all_y_bwd) # Number of unique -y values
        
        # 4. Calculate dp_bwd[i] = length of the longest increasing path starting at coordinates[i].
        # The helper function handles the sorted `aug_coords_bwd` and the `y_map_bwd` correctly.
        # The result dp_bwd[i] corresponds to the longest path starting from original point i.
        dp_bwd = self._calculate_dp(n, aug_coords_bwd, y_map_bwd, m_bwd)
        
        # --- Combine Results ---
        # The longest path that passes through coordinates[k] can be constructed by joining:
        #   - The longest path ending at coordinates[k] (length dp_fwd[k])
        #   - The longest path starting at coordinates[k] (length dp_bwd[k])
        # Since coordinates[k] is included in both paths, it's counted twice. 
        # We subtract 1 to get the correct total length of the combined path.
        result = dp_fwd[k] + dp_bwd[k] - 1
        
        return result