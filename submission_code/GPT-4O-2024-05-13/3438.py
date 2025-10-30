from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def is_peak(arr, i):
            return arr[i] > arr[i-1] and arr[i] > arr[i+1]
        
        result = []
        
        for query in queries:
            if query[0] == 1:
                l, r = query[1], query[2]
                count = 0
                for i in range(l + 1, r):
                    if is_peak(nums, i):
                        count += 1
                result.append(count)
            elif query[0] == 2:
                index, val = query[1], query[2]
                nums[index] = val
        
        return result