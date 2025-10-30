class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        first_occurrence = dict()
        last_occurrence = dict()
        
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
        
        for start, end in intervals:
            if not merged:
                merged.append((start, end))
            else:
                last_start, last_end = merged[-1]
                if start <= last_end:
                    new_start = last_start
                    new_end = max(last_end, end)
                    merged[-1] = (new_start, new_end)
                else:
                    merged.append((start, end))
        
        k = len(merged)
        mod = 10**9 + 7
        return pow(2, k - 1, mod)