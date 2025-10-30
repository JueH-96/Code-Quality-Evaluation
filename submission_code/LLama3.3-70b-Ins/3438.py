from typing import List

class Solution:
    def countOfPeaks(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        def is_peak(nums, i):
            """Check if the element at index i is a peak."""
            return nums[i-1] < nums[i] > nums[i+1]

        result = []
        for query in queries:
            if query[0] == 1:
                # Count the number of peaks in the subarray nums[l..r]
                count = 0
                for i in range(query[1] + 1, query[2]):
                    if is_peak(nums, i):
                        count += 1
                result.append(count)
            elif query[0] == 2:
                # Update the value of nums[index] to val
                nums[query[1]] = query[2]
        return result