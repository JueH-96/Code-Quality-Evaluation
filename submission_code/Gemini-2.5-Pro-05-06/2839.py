import bisect

class Solution:
  def maximumSumQueries(self, nums1: list[int], nums2: list[int], queries: list[list[int]]) -> list[int]:
    n = len(nums1)
    m = len(queries)

    # Create pairs (nums1[j], nums2[j])
    points = []
    for i in range(n):
      points.append((nums1[i], nums2[i]))

    # Sort points by nums1[j] in descending order
    points.sort(key=lambda p: p[0], reverse=True)

    # Augment queries with original indices and sort by x_i in descending order
    indexed_queries = []
    for i in range(m):
      # Assuming standard 0-indexed access for queries[i][0] as x_i and queries[i][1] as y_i
      indexed_queries.append((queries[i][0], queries[i][1], i))
    
    indexed_queries.sort(key=lambda q: q[0], reverse=True)

    # Collect all unique y-coordinates from nums2 values for coordinate compression.
    # These will be the leaves of our segment tree.
    # Using set for unique values, then sort to have a defined order for bisect_left and mapping.
    # It's important that unique_y_coords are from `nums2` values encountered in `points`.
    all_nums2_values = [p[1] for p in points]
    if not all_nums2_values: # Edge case: no points (n=0), though constraints say n >= 1
        return [-1] * m
        
    unique_y_coords = sorted(list(set(all_nums2_values)))
    
    # map_y_to_compressed_idx: Maps an actual y-coordinate value to its compressed 0-based index
    map_y_to_compressed_idx = {y_val: i for i, y_val in enumerate(unique_y_coords)}
    
    K_y = len(unique_y_coords) # Number of unique y-coordinates from nums2

    # Segment Tree Implementation (iterative, using 1-based indexing for tree nodes internally)
    # P is the smallest power of 2 >= K_y. This P is the offset to reach leaf layer.
    # Leaf corresponding to compressed_idx `k` is at seg_tree array index `P + k`.
    P = 1
    while P < K_y:
      P *= 2
    
    # Tree stores maximum sums. Initialize with -1.
    # Size 2*P. Indices 1 to 2*P-1 are used.
    seg_tree = [-1] * (2 * P)

    def update_seg_tree(idx_compressed: int, value: int):
      pos = P + idx_compressed
      
      # Update leaf only if new value is greater.
      # This handles multiple points mapping to same y_compressed_idx, or
      # updates to an existing leaf if a new point offers a better sum.
      if value > seg_tree[pos]:
          seg_tree[pos] = value
      else: # If not greater, no propagation needed for this path.
          return

      # Propagate update upwards
      pos //= 2
      while pos >= 1:
        new_max_for_parent = max(seg_tree[pos * 2], seg_tree[pos * 2 + 1])
        # Optimization: if parent value is unchanged, ancestors also won't change.
        if seg_tree[pos] == new_max_for_parent:
            break
        seg_tree[pos] = new_max_for_parent
        pos //= 2

    def query_seg_tree(min_y_compressed_idx: int) -> int:
      # Query for max sum in range of compressed indices [min_y_compressed_idx, K_y - 1]
      # If min_y_compressed_idx >= K_y, means no y_coord from nums2 satisfies condition (y >= query_y).
      if min_y_compressed_idx >= K_y:
        return -1
      
      # Map to leaf node range in seg_tree array
      # Query range is inclusive: [left_leaf_idx, right_leaf_idx]
      # These are indices in the seg_tree array.
      left = P + min_y_compressed_idx
      right = P + (K_y - 1) 

      max_val = -1
      while left <= right:
        if left % 2 == 1: # If left is a right child, include its value and move to next node over.
          max_val = max(max_val, seg_tree[left])
          left += 1
        if right % 2 == 0: # If right is a left child, include its value and move to previous node over.
          max_val = max(max_val, seg_tree[right])
          right -= 1
        # Current left/right are processed or skipped. Move to parent level.
        left //= 2
        right //= 2
      return max_val

    ans = [-1] * m
    point_idx = 0 # Pointer for iterating through sorted `points`

    for qx, qy, original_idx in indexed_queries:
      # Add points (p_nums1, p_nums2) to segment tree if p_nums1 >= qx
      while point_idx < n and points[point_idx][0] >= qx:
        p_nums1, p_nums2 = points[point_idx]
        p_sum = p_nums1 + p_nums2
        
        compressed_p_nums2_idx = map_y_to_compressed_idx[p_nums2]
        update_seg_tree(compressed_p_nums2_idx, p_sum)
        point_idx += 1
      
      # Query segment tree for y-constraint qy
      # Find smallest compressed_idx `k` such that unique_y_coords[k] >= qy
      # This is what bisect_left does.
      query_min_compressed_y_idx = bisect.bisect_left(unique_y_coords, qy)
      
      current_max_sum = query_seg_tree(query_min_compressed_y_idx)
      ans[original_idx] = current_max_sum
      
    return ans