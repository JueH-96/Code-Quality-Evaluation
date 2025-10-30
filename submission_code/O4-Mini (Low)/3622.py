from typing import List

class Solution:
    def maxFrequency(self, nums: List[int], k: int, numOperations: int) -> int:
        # We build for each i an interval [nums[i]-k, nums[i]+k] of reachable x
        # We'll do a sweep line on these intervals to know at each candidate x
        # how many intervals cover it (cover_count). Also track exact counts.
        
        from collections import Counter
        
        n = len(nums)
        # Frequency of exact nums values
        freq = Counter(nums)
        
        # Build sweep events: +1 at l, -1 at r+1
        events = {}
        for v in nums:
            l = v - k
            r = v + k
            events[l] = events.get(l, 0) + 1
            events[r + 1] = events.get(r + 1, 0) - 1
        
        # Collect all interesting x: event positions and exact-value positions
        keys = set(events.keys()) | set(freq.keys())
        sorted_keys = sorted(keys)
        
        cover = 0
        ans = 0
        # Sweep
        for x in sorted_keys:
            # apply events at x
            cover += events.get(x, 0)
            # exact count at x
            exact = freq.get(x, 0)
            # non-exact reachable = cover - exact
            non_exact = cover - exact
            # we can pick up to numOperations among these non-exact to convert
            use_ops = numOperations if non_exact >= numOperations else non_exact
            # total frequency at x
            cand = exact + use_ops
            if cand > ans:
                ans = cand
        
        return ans