class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        first_occurrence = {}
        last_occurrence = {}
        
        for i, num in enumerate(nums):
            if num not in first_occurrence:
                first_occurrence[num] = i
            last_occurrence[num] = i
        
        intervals = []
        for num in first_occurrence:
            start = first_occurrence[num]
            end = last_occurrence[num]
            intervals.append((start, end))
        
        intervals.sort()
        
        merged = []
        for interval in intervals:
            if not merged:
                merged.append(interval)
            else:
                last_start, last_end = merged[-1]
                curr_start, curr_end = interval
                if curr_start <= last_end:
                    merged[-1] = (last_start, max(last_end, curr_end))
                else:
                    merged.append(interval)
        
        k = len(merged)
        return pow(2, k - 1, MOD) if k > 0 else 0