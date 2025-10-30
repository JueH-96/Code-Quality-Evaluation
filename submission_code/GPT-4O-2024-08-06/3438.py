from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def is_peak(arr, i):
            return arr[i] > arr[i - 1] and arr[i] > arr[i + 1]

        results = []
        
        for query in queries:
            if query[0] == 1:
                # Count peaks in the subarray nums[l_i..r_i]
                l_i, r_i = query[1], query[2]
                peak_count = 0
                for i in range(l_i + 1, r_i):
                    if is_peak(nums, i):
                        peak_count += 1
                results.append(peak_count)
            elif query[0] == 2:
                # Update nums[index_i] to val_i
                index_i, val_i = query[1], query[2]
                nums[index_i] = val_i
        
        return results