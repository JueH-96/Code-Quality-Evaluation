class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        nums.sort()
        left = 0
        right = len(nums) - 1
        while left <= right:
            min_element = nums[left]
            max_element = nums[right]
            average = (min_element + max_element) / 2
            averages.append(average)
            left += 1
            right -= 1
        return min(averages)