from typing import List
class Solution:
    def countSubarrays(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Build mapping value -> list of indices
        pos = {}
        for i, v in enumerate(nums):
            pos.setdefault(v, []).append(i)
        # Prepare linked list for barriers: nodes 0 (virtual -1), 1..n (real 0..n-1), n+1 (virtual n)
        prev = list(range(-1, n+1))  # prev[i] = i-1
        next = list(range(1, n+3))   # next[i] = i+1
        # We'll only use indices 0..n+1 of these arrays
        # Initially no real barriers: only node 0 and node n+1 are ends
        
        def remove_node(u):
            # remove node u from the linked list
            l = prev[u]
            r = next[u]
            next[l] = r
            prev[r] = l
        
        total = 0
        # Process values in descending order; barriers represent indices of nums > current v
        for v in sorted(pos.keys(), reverse=True):
            idxs = pos[v]
            # Group positions by current segment between barriers
            segmap = {}
            for idx in idxs:
                node = idx + 1
                lnode = prev[node]
                rnode = next[node]
                # real boundary values
                seg_start = lnode - 1  # barrier real index
                seg_end = rnode - 1    # barrier real index
                # segment is (seg_start+1 .. seg_end-1) inclusive in real indices
                key = (seg_start, seg_end)
                segmap.setdefault(key, []).append(idx)
            # For each segment, count subarrays with at least k occurrences of v
            for (bstart, bend), a in segmap.items():
                a.sort()
                t = len(a)
                if t < k:
                    continue
                seg_l = bstart + 1
                seg_r = bend - 1
                # sliding window over the k-occurrence window
                for i in range(k-1, t):
                    j = i - (k-1)
                    left_bound = seg_l if j == 0 else a[j-1] + 1
                    left_choices = a[j] - left_bound + 1
                    right_choices = seg_r - a[i] + 1
                    total += left_choices * right_choices
            # Now insert current v's positions as new barriers for next rounds
            for idx in idxs:
                remove_node(idx + 1)
        return total