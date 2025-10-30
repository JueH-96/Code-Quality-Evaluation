from typing import List

class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        nums.sort()  # Sort the list to easily access the smallest and largest elements
        averages = []

        for _ in range(len(nums) // 2):
            min_element = nums.pop(0)  # Remove the smallest element
            max_element = nums.pop()   # Remove the largest element
            averages.append((min_element + max_element) / 2)

        return min(averages)