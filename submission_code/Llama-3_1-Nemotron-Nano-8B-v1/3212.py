from typing import List

class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        
        first_occurrence = {}
        last_occurrence = {}
        
        for i in range(n):
            num = nums[i]
            if num not in first_occurrence:
                first_occurrence[num] = i
            last_occurrence[num] = i
        
        intervals = []
        for num in first_occurrence:
            start = first_occurrence[num]
            end = last_occurrence[num] - 1
            if start <= end:
                intervals.append((start, end))
        
        if not intervals:
            m = n - 1
        else:
            intervals.sort()
            merged = []
            for s, e in intervals:
                if not merged:
                    merged.append((s, e))
                else:
                    last_s, last_e = merged[-1]
                    if s <= last_e:
                        new_s = last_s
                        new_e = max(last_e, e)
                        merged[-1] = (new_s, new_e)
                    else:
                        merged.append((s, e))
            
            sum_length = 0
            for s, e in merged:
                sum_length += e - s + 1
            
            m = (n - 1) - sum_length
        
        return pow(2, m, MOD) if m >= 0 else 0