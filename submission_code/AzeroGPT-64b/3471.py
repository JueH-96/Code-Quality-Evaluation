from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        min_avg = float('inf')

        for i in range(len(nums) // 2):
            # Remove the smallest and largest elements
            min_element = min(nums)
            nums.remove(min_element)
            max_element = max(nums)
            nums.remove(max_element)

            # Calculate the average and update min_avg if necessary
            avg = (min_element + max_element) / 2
            min_avg = min(min_avg, avg)

            # Add the average back to nums
            nums.append(avg)

        return min_avg