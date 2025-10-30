class Solution:
    def maxOperations(self, nums: List[int]) -> int:
        if len(nums) < 2:
            return 0
        
        max_operations = 0
        for i in range(len(nums) - 1):
            score = nums[i] + nums[i + 1]
            operations = 0
            j = 0
            while j < len(nums) - 1:
                if nums[j] + nums[j + 1] == score:
                    operations += 1
                    j += 2
                else:
                    j += 1
            max_operations = max(max_operations, operations)
        
        return max_operations