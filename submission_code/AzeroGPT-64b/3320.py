class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        max_operations = 0
        if len(nums) > 1:
            score = nums[0] + nums[1]
            i = 0
            while i < len(nums)-1:
                if nums[i] + nums[i+1] == score:
                    max_operations += 1
                    i += 2
                else:
                    break
        return max_operations