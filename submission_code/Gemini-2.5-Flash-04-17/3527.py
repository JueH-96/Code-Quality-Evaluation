from sortedcontainers import SortedList
from typing import List

class FenwickTree:
    def __init__(self, size):
        self.size = size
        # Fenwick tree is 1-indexed internally, size + 1 array
        self.tree = [0] * (size + 1)

    def update(self, idx, delta):
        # Update at 0-indexed idx
        idx += 1 # Convert to 1-indexed
        while idx <= self.size:
            self.tree[idx] += delta
            idx += idx & (-idx)

    def query(self, idx):
        # Query sum up to 0-indexed idx
        idx += 1 # Convert to 1-indexed
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & (-idx)
        return res

class Solution:
    def numberOfAlternatingGroups(self, colors: List[int], queries: List[List[int]]) -> List[int]:
        n = len(colors)
        
        # is_alt[i] = 1 if colors[i] != colors[(i+1)%n], 0 otherwise
        is_alt = [0] * n
        for i in range(n):
            is_alt[i] = 1 if colors[i] != colors[(i+1)%n] else 0

        # Store indices where is_alt is 0 (the "breaks")
        breaks = SortedList()
        for i in range(n):
            if is_alt[i] == 0:
                breaks.add(i)

        # Fenwick trees to maintain counts and sum of values for differences between breaks
        # Differences D are positive integers from 1 to n. FT size n. Use 0-indexed FT array internally, map diff D to index D-1.
        # Max diff is n, index n-1. FT size needed is n.
        ft_count = FenwickTree(n) # Store counts of differences D (mapped to index D-1)
        ft_sum_val = FenwickTree(n) # Store sum of value * count for differences D (mapped to index D-1)

        # Helper function to update FTs with a difference value
        def update_diff_ft(diff, delta_count):
            if diff <= 0: return # Should not happen with diffs between breaks unless n < 1 (constraint n >= 4)
            # Map diff value D to FT index D-1 (0-indexed)
            ft_count.update(diff - 1, delta_count)
            ft_sum_val.update(diff - 1, delta_count * diff)

        # Helper to get prefix sum of counts up to diff_max (inclusive, diff value)
        def query_ft_count_le(diff_max_val):
            if diff_max_val < 1: return 0
            if diff_max_val >= n: return ft_count.query(n - 1) # Query all diffs up to n
            return ft_count.query(diff_max_val - 1) # Query up to index diff_max_val - 1

        # Helper to get prefix sum of value * count up to diff_max (inclusive, diff value)
        def query_ft_sum_val_le(diff_max_val):
            if diff_max_val < 1: return 0
            if diff_max_val >= n: return ft_sum_val.query(n - 1) # Query all diffs up to n
            return ft_sum_val.query(diff_max_val - 1) # Query up to index diff_max_val - 1

        # Helper to get previous break index circularly from the current breaks list
        def get_prev_break(idx):
            # Assumes breaks list is not empty
            i = breaks.bisect_left(idx)
            if i == 0: return breaks[-1] # Wrap around
            return breaks[i-1]
            
        # Helper to get next break index circularly from the current breaks list
        def get_next_break(idx):
            # Assumes breaks list is not empty
            i = breaks.bisect_right(idx)
            if i == len(breaks): return breaks[0] # Wrap around
            return breaks[i]

        # Initial population of FTs
        m = len(breaks)
        if 0 < m < n:
            for i in range(m):
                b_curr = breaks[i]
                b_next = breaks[(i + 1) % m]
                diff = (b_next - b_curr + n) % n
                update_diff_ft(diff, 1)

        results = []

        for query in queries:
            type = query[0]

            if type == 1:
                size = query[1] # k
                
                m = len(breaks)
                
                if m == 0: # All 1s (All alternating)
                    # If is_alt is all 1s, any k-sized contiguous group is alternating.
                    # There are n such groups (starting at each position).
                    # Constraint: 3 <= size <= colors.length - 1
                    results.append(n)
                elif m == n: # All 0s (No alternating groups of size >= 2)
                    results.append(0)
                else:
                    # We need sum over all differences D_j = length_of_1s_segment + 1
                    # of max(0, (D_j - 1) - (size - 1) + 1) = max(0, D_j - size + 1)
                    # This is sum_{D >= size} (D - size + 1) where D is difference between breaks (>= 1)
                    
                    # sum_{D >= size} 1 = total count - sum_{D < size} 1 = total count - sum_{D <= size-1} 1
                    count_ge_size = query_ft_count_le(n) - query_ft_count_le(size - 1) 
                    
                    # sum_{D >= size} D = total sum of D - sum_{D < size} D = total sum of D - sum_{D <= size-1} D
                    sum_val_ge_size = query_ft_sum_val_le(n) - query_ft_sum_val_le(size - 1)
                    
                    count = sum_val_ge_size - (size - 1) * count_ge_size
                    results.append(count)

            else: # type == 2, Update
                idx = query[1]
                new_color = query[2]
                
                old_color = colors[idx]
                if old_color == new_color:
                    continue # No change

                # Indices in is_alt array affected are idx and (idx-1+n)%n
                idx1 = idx
                idx2 = (idx - 1 + n) % n
                
                # Store old is_alt values before updating colors
                old_v1 = is_alt[idx1]
                old_v2 = is_alt[idx2] # is_alt[idx2] depends on colors[idx2] and colors[idx1]

                # Update colors array
                colors[idx] = new_color
                
                # Calculate new is_alt values based on updated colors
                new_v1 = 1 if colors[idx1] != colors[(idx1 + 1) % n] else 0
                new_v2 = 1 if colors[idx2] != colors[(idx2 + 1) % n] else 0

                # Process change at idx1
                if old_v1 != new_v1:
                    if new_v1 == 1: # 0 -> 1 at idx1: idx1 removed from breaks
                        m_before = len(breaks)
                        if m_before > 0: # Should be true if old_v1 was 0
                            p_b = get_prev_break(idx1)
                            q_b = get_next_break(idx1)
                            if m_before == 1: # idx1 must be the only break
                                update_diff_ft(n, -1) # Remove diff n
                            else: # idx1 was one of multiple breaks
                                # The single old diff was (q_b - p_b + n)%n
                                old_diff_val = (q_b - p_b + n) % n
                                update_diff_ft(old_diff_val, -1)
                                # The two new diffs are (idx1 - p_b + n)%n and (q_b - idx1 + n)%n
                                new_diff_val1 = (idx1 - p_b + n) % n
                                new_diff_val2 = (q_b - idx1 + n) % n
                                update_diff_ft(new_diff_val1, 1)
                                update_diff_ft(new_diff_val2, 1)
                        breaks.remove(idx1)
                        is_alt[idx1] = 1
                    else: # new_v1 == 0 (1 -> 0 at idx1): idx1 added to breaks
                        # Add idx1 to breaks temporarily to find neighbors
                        breaks.add(idx1) 
                        m_after = len(breaks)
                        if m_after == 1: # idx1 must be the only break now
                             update_diff_ft(n, 1) # Add diff n
                        else: # idx1 is one of multiple breaks now
                            p_b = get_prev_break(idx1)
                            q_b = get_next_break(idx1)
                            # The two old diffs were (idx1 - p_b + n)%n and (q_b - idx1 + n)%n
                            old_diff_val1 = (idx1 - p_b + n) % n
                            old_diff_val2 = (q_b - idx1 + n) % n
                            update_diff_ft(old_diff_val1, -1)
                            update_diff_ft(old_diff_val2, -1)
                            # The single new diff is (q_b - p_b + n)%n
                            new_diff_val = (q_b - p_b + n) % n
                            update_diff_ft(new_diff_val, 1)
                        # Note: idx1 was already added to breaks.
                        is_alt[idx1] = 0

                # Process change at idx2
                # Note: breaks and FTs are already updated if idx1 flipped.
                if idx1 != idx2 and old_v2 != new_v2: # idx1 == idx2 only if n=1, not possible
                    if new_v2 == 1: # 0 -> 1 at idx2: idx2 removed from breaks
                         m_before = len(breaks)
                         if m_before > 0: # Should be true if old_v2 was 0
                            p_b = get_prev_break(idx2)
                            q_b = get_next_break(idx2)
                            if m_before == 1: # idx2 must be the only break
                                update_diff_ft(n, -1)
                            else: # idx2 was one of multiple breaks
                                old_diff_val = (q_b - p_b + n) % n
                                update_diff_ft(old_diff_val, -1)
                                new_diff_val1 = (idx2 - p_b + n) % n
                                new_diff_val2 = (q_b - idx2 + n) % n
                                update_diff_ft(new_diff_val1, 1)
                                update_diff_ft(new_diff_val2, 1)
                         breaks.remove(idx2)
                         is_alt[idx2] = 1
                    else: # new_v2 == 0 (1 -> 0 at idx2): idx2 added to breaks
                        # Add idx2 to breaks temporarily to find neighbors
                        breaks.add(idx2) 
                        m_after = len(breaks)
                        if m_after == 1: # idx2 must be the only break now
                            update_diff_ft(n, 1)
                        else: # idx2 is one of multiple breaks now
                            p_b = get_prev_break(idx2)
                            q_b = get_next_break(idx2)
                            old_diff_val1 = (idx2 - p_b + n) % n
                            old_diff_val2 = (q_b - idx2 + n) % n
                            update_diff_ft(old_diff_val1, -1)
                            update_diff_ft(old_diff_val2, -1)
                            new_diff_val = (q_b - p_b + n) % n
                            update_diff_ft(new_diff_val, 1)
                        # Note: idx2 was already added to breaks.
                        is_alt[idx2] = 0
        return results