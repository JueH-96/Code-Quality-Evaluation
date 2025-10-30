from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def is_peak(i):
            return nums[i] > nums[i - 1] and nums[i] > nums[i + 1]
        
        def update_peak_count(index, val):
            if index > 0 and index < len(nums) - 1:
                if is_peak(index):
                    peak_count[index] -= 1
                if is_peak(index - 1):
                    peak_count[index - 1] -= 1
                if is_peak(index + 1):
                    peak_count[index + 1] -= 1
                nums[index] = val
                if is_peak(index):
                    peak_count[index] += 1
                if is_peak(index - 1):
                    peak_count[index - 1] += 1
                if is_peak(index + 1):
                    peak_count[index + 1] += 1
            else:
                nums[index] = val
        
        def count_peaks(l, r):
            return sum(peak_count[i] for i in range(l + 1, r) if peak_count[i] > 0)
        
        peak_count = [0] * len(nums)
        for i in range(1, len(nums) - 1):
            if is_peak(i):
                peak_count[i] = 1
        
        result = []
        for query in queries:
            if query[0] == 1:
                l, r = query[1], query[2]
                result.append(count_peaks(l, r + 1))
            elif query[0] == 2:
                index, val = query[1], query[2]
                update_peak_count(index, val)
        
        return result