from sortedcontainers import SortedList
from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        N = len(nums)
        peak_indices = SortedList()
        
        # Initialize peak_indices
        for i in range(1, N-1):
            if nums[i] > nums[i-1] and nums[i] > nums[i+1]:
                peak_indices.add(i)
        
        answer = []
        
        for query in queries:
            if query[0] == 1:
                l, r = query[1], query[2]
                if r - l + 1 < 3:
                    answer.append(0)
                else:
                    count = peak_indices.bisect_right(r - 1) - peak_indices.bisect_left(l + 1)
                    answer.append(count)
            else:
                index_i, val_i = query[1], query[2]
                old_val = nums[index_i]
                nums[index_i] = val_i
                
                # Check and update peak status for index_i-1, index_i, index_i+1
                for idx in [index_i - 1, index_i, index_i + 1]:
                    if 1 <= idx <= N - 2:
                        if nums[idx] > nums[idx - 1] and nums[idx] > nums[idx + 1]:
                            peak_indices.add(idx)
                        else:
                            peak_indices.discard(idx)
        
        return answer