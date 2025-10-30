import math

class Solution:
    def minimumMoves(self, nums: list[int], k: int, maxChanges: int) -> int:
        if k == 0:
            return 0

        pos = [i for i, x in enumerate(nums) if x == 1]
        m = len(pos)

        ans = float('inf')

        # This part handles cases where we rely heavily on `maxChanges`.
        # These are simple but powerful scenarios that give good initial bounds for the answer.
        # Cost to gather 1 existing `1` and `k-1` created ones:
        if m >= 1 and k - 1 <= maxChanges:
            ans = min(ans, (k-1) * 2)
        
        # Cost to gather 2 closest existing `1`s and `k-2` created ones:
        if m >= 2 and k - 2 <= maxChanges:
            min_dist_2 = float('inf')
            for i in range(m - 1):
                min_dist_2 = min(min_dist_2, pos[i+1] - pos[i])
            ans = min(ans, min_dist_2 + (k-2) * 2)
            
        # Cost to gather 3 closest existing `1`s and `k-3` created ones:
        if m >= 3 and k - 3 <= maxChanges:
            min_dist_3 = float('inf')
            for i in range(m - 2):
                min_dist_3 = min(min_dist_3, pos[i+2] - pos[i])
            ans = min(ans, min_dist_3 + (k-3) * 2)

        # The core logic: an optimal solution will only use existing `1`s that are "close" to each other.
        # "Close" means the distance between consecutive `1`s is at most 2.
        # We partition `pos` into groups of such close `1`s.
        close_ones_groups = []
        if m > 0:
            current_group = [pos[0]]
            for i in range(m - 1):
                if pos[i+1] - pos[i] <= 2:
                    current_group.append(pos[i+1])
                else:
                    close_ones_groups.append(current_group)
                    current_group = [pos[i+1]]
            close_ones_groups.append(current_group)

        # Precompute prefix sums for each group for efficient cost calculation.
        prefix_sums_map = {}
        for g in close_ones_groups:
            if not g: continue
            ps = [0] * (len(g) + 1)
            for i in range(len(g)):
                ps[i+1] = ps[i] + g[i]
            prefix_sums_map[id(g)] = ps
        
        for group in close_ones_groups:
            group_len = len(group)
            if group_len == 0: continue
            group_ps = prefix_sums_map[id(group)]
            
            # We can use `w` ones from this group and `c=k-w` changes.
            # The number of changes `c` must be at most `maxChanges`.
            # This implies `w >= k - maxChanges`.
            w_min = max(1, k - maxChanges)
            
            for w in range(w_min, min(k, group_len) + 1):
                # We find the minimum cost to gather a sub-group of size `w` from the current group.
                # This is a sliding window of size `w` on the group.
                for i in range(group_len - w + 1):
                    # window is group[i...i+w-1]
                    median_idx_in_group = i + (w - 1) // 2
                    median_val = group[median_idx_in_group]
                    
                    # Cost = sum |x - median| for x in the window.
                    # This can be calculated in O(1) with prefix sums.
                    left_count = median_idx_in_group - i
                    right_count = (i + w - 1) - median_idx_in_group
                    
                    sum_left = group_ps[median_idx_in_group] - group_ps[i]
                    sum_right = group_ps[i+w] - group_ps[median_idx_in_group + 1]
                    
                    gather_cost = (left_count * median_val - sum_left) + \
                                  (sum_right - right_count * median_val)
                    
                    total_cost = gather_cost + 2 * (k - w)
                    ans = min(ans, total_cost)

        return ans if ans != float('inf') else k * 2