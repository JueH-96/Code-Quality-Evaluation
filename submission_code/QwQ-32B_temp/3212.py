class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        first = {}
        last = {}
        for i, num in enumerate(nums):
            if num not in first:
                first[num] = i
            last[num] = i
        
        intervals = []
        for num in first:
            intervals.append((first[num], last[num]))
        
        if not intervals:
            return 0  # This case shouldn't happen as nums is non-empty
        
        intervals.sort()
        merged = []
        current_start, current_end = intervals[0]
        for s, e in intervals[1:]:
            if s <= current_end:
                current_end = max(current_end, e)
            else:
                merged.append((current_start, current_end))
                current_start, current_end = s, e
        merged.append((current_start, current_end))
        
        m = len(merged)
        return pow(2, m - 1, MOD)