class Solution:
    def minimumAverage(self, nums: List[int]) -> float:
        nums.sort()
        averages = []
        left, right = 0, len(nums)-1
        
        while left < right:
            avg = (nums[left] + nums[right]) / 2
            averages.append(avg)
            left += 1
            right -= 1
        
        return min(averages)