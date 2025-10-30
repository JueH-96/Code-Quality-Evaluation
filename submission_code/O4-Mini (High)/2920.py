class Solution:
    def minimumSeconds(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        n = len(nums)
        # record the indices of each value
        pos = defaultdict(list)
        for i, v in enumerate(nums):
            pos[v].append(i)
        
        ans = float('inf')
        # for each distinct value, compute how long it takes to flood the whole ring
        for positions in pos.values():
            # positions are in ascending order by construction
            max_gap = 0
            # gaps between consecutive occurrences
            for i in range(len(positions) - 1):
                gap = positions[i+1] - positions[i] - 1
                if gap > max_gap:
                    max_gap = gap
            # wrap‑around gap from last occurrence back to first
            wrap_gap = positions[0] + n - positions[-1] - 1
            if wrap_gap > max_gap:
                max_gap = wrap_gap
            # time to fill a gap of length max_gap is ceil(max_gap/2)
            t = (max_gap + 1) // 2
            ans = min(ans, t)
        
        return ans