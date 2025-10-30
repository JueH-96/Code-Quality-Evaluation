class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        first_occurrence = dict()
        last_occurrence = dict()
        
        for i, num in enumerate(nums):
            if num not in first_occurrence:
                first_occurrence[num] = i
            last_occurrence[num] = i
        
        intervals = []
        for num in first_occurrence:
            intervals.append((first_occurrence[num], last_occurrence[num]))
        
        intervals.sort()
        merged = []
        
        for interval in intervals:
            if not merged:
                merged.append(interval)
            else:
                prev_start, prev_end = merged[-1]
                curr_start, curr_end = interval
                if curr_start <= prev_end:
                    # Merge the intervals
                    merged[-1] = (prev_start, max(prev_end, curr_end))
                else:
                    merged.append(interval)
        
        m = len(merged)
        return pow(2, m - 1, MOD)