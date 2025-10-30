import collections
# This problem typically requires a SortedList or a custom balanced BST/Skip List
# for efficient operations on `zero_indices`. Assuming `sortedcontainers` is available
# as is common in competitive programming environments on platforms like LeetCode.
from sortedcontainers import SortedList

class FenwickTree:
    """
    A Fenwick Tree (Binary Indexed Tree) for point updates and prefix sum queries.
    0-indexed for external interface (update and query parameters).
    1-indexed internally for tree array.
    """
    def __init__(self, size):
        # Tree size is size + 1 because it's 1-indexed, and max index is `size`.
        self.tree = [0] * (size + 1)
        self.size = size

    def update(self, index, delta):
        # Update value at 'index' by 'delta'.
        # Convert 0-based index to 1-based for BIT.
        index += 1
        while index <= self.size:
            self.tree[index] += delta
            index += index & (-index)

    def query(self, index):
        # Query sum from 0 to 'index' (inclusive).
        # Convert 0-based index to 1-based for BIT.
        index += 1
        s = 0
        while index > 0:
            s += self.tree[index]
            index -= index & (-index)
        return s

    def query_range(self, l, r):
        # Query sum from 'l' to 'r' (inclusive).
        # Handles cases where l > r, returning 0.
        if l > r:
            return 0
        return self.query(r) - self.query(l - 1)


class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        N = len(colors)
        
        # is_alt[i] = 1 if colors[i] != colors[(i+1)%N], else 0
        is_alt = [0] * N
        for i in range(N):
            if colors[i] != colors[(i+1)%N]:
                is_alt[i] = 1
            else:
                is_alt[i] = 0
        
        # Maintain indices of 0s in `is_alt` using SortedList for O(log N) operations.
        zero_indices = SortedList()
        for i in range(N):
            if is_alt[i] == 0:
                zero_indices.add(i)

        # Fenwick trees for managing segment lengths and their counts.
        # Max length of a '1' segment can be N (if all `is_alt` are 1s).
        bit_counts = FenwickTree(N) # Stores counts of segments of a particular length
        bit_sums = FenwickTree(N)   # Stores sum of lengths (length * count)

        # --- Initial population of Fenwick trees based on initial `is_alt` segments ---
        if not zero_indices: # All `is_alt` values are 1s
            if N > 0: # A segment of length N is present
                bit_counts.update(N, 1)
                bit_sums.update(N, N)
        else:
            # Iterate through segments of 1s defined by consecutive 0s in `zero_indices`.
            # Handle circularity by considering the segment between the last zero and the first zero.
            
            # The indices in zero_indices are sorted. `z_k` is last, `z_1` is first.
            # Segment between `z_k` and `z_1` (circularly)
            last_zero_idx = zero_indices[-1]
            first_zero_idx_ext = zero_indices[0] + N 
            
            circular_segment_len = first_zero_idx_ext - last_zero_idx - 1
            if circular_segment_len > 0:
                bit_counts.update(circular_segment_len, 1)
                bit_sums.update(circular_segment_len, circular_segment_len)
            
            # Segments between linear consecutive zeros (z_i to z_{i+1})
            for i in range(len(zero_indices) - 1):
                z_curr = zero_indices[i]
                z_next = zero_indices[i+1]
                linear_segment_len = z_next - z_curr - 1
                if linear_segment_len > 0:
                    bit_counts.update(linear_segment_len, 1)
                    bit_sums.update(linear_segment_len, linear_segment_len)

        results = []
        for query in queries:
            query_type = query[0]

            if query_type == 1: # Count alternating groups of size size_i
                size_i = query[1]
                L = size_i - 1 # An alternating group of size k means k-1 consecutive 1s in `is_alt`.
                
                # We need to sum (segment_length - L + 1) for all segments where segment_length >= L.
                # This is equivalent to sum(segment_length) - (L - 1) * sum(1).
                total_count_ge_L = bit_counts.query_range(L, N)
                total_sum_lengths_ge_L = bit_sums.query_range(L, N)
                
                ans = total_sum_lengths_ge_L - (L - 1) * total_count_ge_L
                results.append(ans)

            else: # Change colors[index_i] to color_i
                index_i, color_i = query[1], query[2]
                
                # If color doesn't change, no `is_alt` values will change.
                if colors[index_i] == color_i:
                    continue 
                
                # Update color array first. This will affect `is_alt` values based on new neighbors.
                colors[index_i] = color_i

                # The change in `colors[index_i]` can affect two `is_alt` entries:
                # 1. `is_alt[index_i]` (compares `colors[index_i]` with `colors[(index_i+1)%N]`)
                # 2. `is_alt[(index_i - 1 + N) % N]` (compares `colors[(index_i - 1 + N)%N]` with `colors[index_i]`)
                
                # Collect the indices in `is_alt` that need to be re-evaluated
                affected_is_alt_indices = [index_i, (index_i - 1 + N) % N]
                
                for current_is_alt_idx in affected_is_alt_indices:
                    # Get the value of this `is_alt` position BEFORE the `colors[index_i]` was changed,
                    # but `is_alt` array has its value from previous state.
                    old_is_alt_value_at_pos = is_alt[current_is_alt_idx]
                    
                    # Calculate the new `is_alt` value at this position using the UPDATED `colors` array.
                    new_is_alt_value_at_pos = 1 if colors[current_is_alt_idx] != colors[(current_is_alt_idx + 1) % N] else 0

                    if old_is_alt_value_at_pos == new_is_alt_value_at_pos:
                        continue # No actual change at this `is_alt` position, skip updates.

                    # Update the `is_alt` array with the new value
                    is_alt[current_is_alt_idx] = new_is_alt_value_at_pos
                    
                    # --- Update `zero_indices` and Fenwick Trees based on `is_alt` change ---
                    if new_is_alt_value_at_pos == 0: # `is_alt[current_is_alt_idx]` changed from 1 to 0
                        # A segment of 1s is being split by a new 0.
                        # This `current_is_alt_idx` needs to be added to `zero_indices`.
                        
                        if not zero_indices: # All `is_alt` were 1s, now one 0 appears.
                            # The single segment of length N is removed, and a segment of N-1 (if > 0) replaces it.
                            bit_counts.update(N, -1)
                            bit_sums.update(N, -N)
                            
                            if N - 1 > 0:
                                bit_counts.update(N - 1, 1)
                                bit_sums.update(N - 1, N - 1)
                            zero_indices.add(current_is_alt_idx)
                        else:
                            # Find the two zeros that are now on either side of `current_is_alt_idx`
                            # (considering circularity). These define the segment of 1s that splits.
                            
                            # Find index in `zero_indices` where `current_is_alt_idx` would be inserted.
                            insert_pos = zero_indices.bisect_left(current_is_alt_idx)
                            
                            # Get circular previous and next zeros.
                            z_prev_idx = zero_indices[insert_pos - 1] if insert_pos > 0 else zero_indices[-1]
                            z_next_idx = zero_indices[insert_pos] if insert_pos < len(zero_indices) else zero_indices[0]
                            
                            # The length of the segment of 1s that is being split
                            old_segment_len = (z_next_idx - z_prev_idx - 1 + N) % N
                            if old_segment_len > 0:
                                bit_counts.update(old_segment_len, -1)
                                bit_sums.update(old_segment_len, -old_segment_len)
                            
                            # Add `current_is_alt_idx` to `zero_indices`
                            zero_indices.add(current_is_alt_idx)
                            
                            # Calculate lengths of the two new segments formed by the split
                            new_segment_len_1 = (current_is_alt_idx - z_prev_idx - 1 + N) % N
                            new_segment_len_2 = (z_next_idx - current_is_alt_idx - 1 + N) % N

                            if new_segment_len_1 > 0:
                                bit_counts.update(new_segment_len_1, 1)
                                bit_sums.update(new_segment_len_1, new_segment_len_1)
                            if new_segment_len_2 > 0:
                                bit_counts.update(new_segment_len_2, 1)
                                bit_sums.update(new_segment_len_2, new_segment_len_2)

                    else: # `new_is_alt_value_at_pos == 1` (`is_alt[current_is_alt_idx]` changed from 0 to 1)
                        # A 0 is being removed, causing two segments of 1s to merge, or extending one segment.
                        # This `current_is_alt_idx` must have been in `zero_indices`. Remove it.
                        zero_indices.remove(current_is_alt_idx)

                        if not zero_indices: # All `is_alt` are now 1s (current_is_alt_idx was the only 0)
                            # Remove the old (N-1) length segment (if > 0) and add the new N length segment.
                            if N - 1 > 0:
                                bit_counts.update(N - 1, -1)
                                bit_sums.update(N - 1, -(N - 1))
                            bit_counts.update(N, 1)
                            bit_sums.update(N, N)
                        else:
                            # Find the two zeros that are now adjacent after `current_is_alt_idx` is removed.
                            # These define the merged segment.
                            
                            # Find insertion point if `current_is_alt_idx` was still there (for neighbors).
                            # Since it's removed, we get its neighbors from the remaining sorted list.
                            
                            # Its old neighbors in zero_indices will now be neighbors of each other.
                            # Example: If `zero_indices` was `[z_prev, current_is_alt_idx, z_next]`
                            # After removal, `z_prev` and `z_next` are now neighbors.
                            
                            insert_pos_if_present = zero_indices.bisect_left(current_is_alt_idx)
                            
                            z_prev_idx = zero_indices[insert_pos_if_present - 1] if insert_pos_if_present > 0 else zero_indices[-1]
                            z_next_idx = zero_indices[insert_pos_if_present] if insert_pos_if_present < len(zero_indices) else zero_indices[0]
                            
                            # The two segments that existed before (separated by current_is_alt_idx, which was 0)
                            # Length of segment from z_prev_idx to current_is_alt_idx (exclusive of current_is_alt_idx)
                            len_before_merge_1 = (current_is_alt_idx - z_prev_idx - 1 + N) % N
                            # Length of segment from current_is_alt_idx to z_next_idx (exclusive of current_is_alt_idx)
                            len_before_merge_2 = (z_next_idx - current_is_alt_idx - 1 + N) % N
                            
                            # Remove these two segments from BITs
                            if len_before_merge_1 > 0:
                                bit_counts.update(len_before_merge_1, -1)
                                bit_sums.update(len_before_merge_1, -len_before_merge_1)
                            if len_before_merge_2 > 0:
                                bit_counts.update(len_before_merge_2, -1)
                                bit_sums.update(len_before_merge_2, -len_before_merge_2)

                            # New merged segment length
                            new_merged_len = (z_next_idx - z_prev_idx - 1 + N) % N
                            if new_merged_len > 0:
                                bit_counts.update(new_merged_len, 1)
                                bit_sums.update(new_merged_len, new_merged_len)
        return results