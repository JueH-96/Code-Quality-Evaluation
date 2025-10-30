class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        """
        We want the number of subarrays in which the subarray's maximum element
        occurs at least k times in that subarray.

        Key idea:
         1) A subarray's maximum is X if and only if the subarray contains no element > X,
            and it contains at least one occurrence of X.
            In our problem, we also require that X appears at least k times in that subarray.
         2) Thus, to count subarrays whose maximum is a given value X occurring >= k times,
            we only look at contiguous segments of the array that do not contain an element > X.
         3) Within such a segment [start..end], let the positions of X be p1 < p2 < ... < pM.
            Then the number of subarrays that contain at least k occurrences of X is calculated
            by the standard "sliding window over positions" technique:
               - Define p0 = start-1, p(M+1) = end+1 for convenience.
               - For each i in [1..M-k+1], subarrays that contain p_i..p_{i+k-1} as a block
                 can choose a left boundary in [p_{i-1}+1..p_i],
                 and a right boundary in [p_{i+k-1}..p_{i+k}-1].
                 That yields (p_i - p_{i-1]) * (p_{i+k} - p_{i+k-1]) subarrays.
            Sum over i. 
         4) We do the above for all distinct values X. However, we must ensure we only count
            subarrays that do NOT contain elements > X. So we process values in descending order:
              - "Remove" from the array the indices of any element > current X (so those positions
                split the array into segments where no element is bigger than X).
              - In each resulting segment, count how many subarrays have maximum = X (i.e. contain
                at least k occurrences of X).
            Because a subarray cannot simultaneously have max = X and max = Y > X, we will
            not double-count. Each valid subarray is counted exactly once, at the step for its max.

        Implementation detail:
          - We sort all element-index pairs by value descending.
          - We keep a data structure ("segments") of active segments where no value > current X.
            Each time we move to a smaller X, we remove the indices where the (already visited)
            values were > X. That removal splits segments. In Python, we can manage this with
            a balanced-tree-like approach or a sorted dictionary of segments.
          - Then within each active segment, we locate the positions of X (via pos_map[X]) 
            and apply the "count sub-subarrays with >= k occurrences" formula.
          - Complexity is roughly O(n log n) if implemented with care.

        Let's implement it.
        """

        import sys
        import bisect

        # Edge case: if k == 1, then every subarray that contains its maximum at least once
        # is just the total number of subarrays minus those that don't contain that element.
        # But we proceed with the general solution (works for k=1 too).

        n = len(nums)
        if k > n:
            return 0

        # 1) Collect positions of each distinct value.
        from collections import defaultdict
        pos_map = defaultdict(list)
        for i, val in enumerate(nums):
            pos_map[val].append(i)

        # 2) Sort the distinct values in descending order.
        distinct_vals = sorted(pos_map.keys(), reverse=True)

        # 3) We'll have all (value, index) pairs in descending order by value.
        val_index_pairs = sorted([(v, i) for i, v in enumerate(nums)], key=lambda x: x[0], reverse=True)

        # Data structure for active segments:
        # We'll store them as a dict: {start: (start, end)}, sorted by start.
        # Initially, the whole array [0..n-1] is active (before removing anything).
        import bisect

        segments = {0: (0, n-1)}  # key = segment_start -> (segment_start, segment_end)
        segment_starts = [0]      # a sorted list of current segment starts

        def find_segment(idx):
            """
            Find which segment currently contains 'idx' using binary search over segment_starts.
            Returns (start, end) if found, or None if 'idx' is not in any segment (already removed).
            """
            # Rightmost position in segment_starts not greater than idx
            # i.e. largest s.t. segment_starts[pos] <= idx
            pos = bisect.bisect_right(segment_starts, idx) - 1
            if pos < 0:
                return None
            start_key = segment_starts[pos]
            s, e = segments[start_key]
            if s <= idx <= e:
                return (s, e)
            return None

        def remove_index(idx):
            """
            Removes index 'idx' from the active segments by splitting.
            If idx is already not in any segment, do nothing.
            """
            seg = find_segment(idx)
            if seg is None:
                return
            (s, e) = seg
            # Remove the old segment from data structures
            del segments[s]
            seg_pos = bisect.bisect_left(segment_starts, s)
            if seg_pos < len(segment_starts) and segment_starts[seg_pos] == s:
                segment_starts.pop(seg_pos)

            # We'll form up to two new segments [s..idx-1] and [idx+1..e], if valid
            if s <= idx-1 >= s:
                segments[s] = (s, idx-1)
                bisect.insort(segment_starts, s)
            if idx+1 <= e:
                segments[idx+1] = (idx+1, e)
                bisect.insort(segment_starts, idx+1)

        ans = 0
        p = 0  # pointer that scans val_index_pairs
        total_pairs = len(val_index_pairs)

        # 4) Iterate over distinct values v in descending order
        for v in distinct_vals:
            # Remove from the array all positions whose value > v,
            # i.e. while val_index_pairs[p].value > v
            while p < total_pairs and val_index_pairs[p][0] > v:
                _, idx_to_remove = val_index_pairs[p]
                remove_index(idx_to_remove)
                p += 1

            # Now the active segments contain no value > v,
            # so subarrays inside those segments can have maximum <= v.
            # Among them, subarrays that have maximum exactly v must contain
            # at least one occurrence of v. We want those with >= k occurrences of v.
            # For each segment, gather positions of v, apply the formula.
            v_positions = pos_map[v]
            # We'll process them in ascending order and segment by segment
            # to count how many fall within each active segment.
            if not v_positions:
                continue

            # We iterate over each active segment, find the sub-range of v_positions
            # that fall into that segment, and use the counting formula.
            # segment_starts is sorted ascending
            for seg_start in segment_starts:
                s, e = segments[seg_start]
                # positions of v in [s..e]
                left_idx = bisect.bisect_left(v_positions, s)
                right_idx = bisect.bisect_right(v_positions, e)
                count_in_seg = right_idx - left_idx
                if count_in_seg < k:
                    # not enough v's to form a subarray with >= k occurrences of v
                    continue

                # Let the relevant v-positions in this segment be p1..pM
                these_pos = v_positions[left_idx:right_idx]
                # We'll use p0 = s-1, p_{M+1} = e+1
                # Then for each i in [1..M-k+1], add
                #    (p_i - p_{i-1]) * (p_{i+k} - p_{i+k-1])
                # to the result.
                # We'll just do a loop.

                # define "extended" positions
                extended_positions = [s-1] + these_pos + [e+1]
                M = len(these_pos)
                for i in range(1, M - k + 2):
                    left_gap = extended_positions[i] - extended_positions[i-1]
                    right_gap = extended_positions[i+k] - extended_positions[i+k-1]
                    ans += left_gap * right_gap

        return ans