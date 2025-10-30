class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        averages = []
        n = len(nums)
        
        while n > 0:
            nums.sort()
            min_element = nums[0]
            max_element = nums[-1]
            averages.append((min_element + max_element) / 2)
            nums.pop(0)
            nums.pop(-1)
            n -= 2
        
        return min(averages)