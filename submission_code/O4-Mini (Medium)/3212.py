class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        # Record first and last occurrence of each value
        first = {}
        last = {}
        for i, x in enumerate(nums):
            if x not in first:
                first[x] = i
            last[x] = i
        
        # Build intervals of forbidden cut positions [l, r]
        # A cut at position i (between i and i+1) is forbidden
        # if it lies between any two occurrences of the same number.
        intervals = []
        for x in first:
            l = first[x]
            r = last[x]
            if r > l:
                # forbid cuts from l .. r-1 inclusive
                intervals.append((l, r - 1))
        
        # Merge the intervals and count total forbidden positions
        intervals.sort()
        forbidden = 0
        cur_start = -1
        cur_end = -1
        for s, e in intervals:
            if cur_start == -1:
                # first interval
                cur_start, cur_end = s, e
            elif s > cur_end + 1:
                # disjoint, accumulate previous
                forbidden += (cur_end - cur_start + 1)
                cur_start, cur_end = s, e
            else:
                # overlap or adjacent, extend
                cur_end = max(cur_end, e)
        # add the last one
        if cur_start != -1:
            forbidden += (cur_end - cur_start + 1)
        
        # total possible cut positions = n-1
        allowed = (n - 1) - forbidden
        if allowed < 0:
            allowed = 0
        
        # Each allowed cut can be present or not => 2^allowed
        return pow(2, allowed, MOD)