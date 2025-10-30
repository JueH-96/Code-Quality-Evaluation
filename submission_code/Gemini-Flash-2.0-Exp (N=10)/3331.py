class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        operations = 0
        while True:
            all_greater_or_equal = True
            for num in nums:
                if num < k:
                    all_greater_or_equal = False
                    break
            if all_greater_or_equal:
                return operations
            
            min_val = float('inf')
            min_index = -1
            for i, num in enumerate(nums):
                if num < min_val:
                    min_val = num
                    min_index = i
            
            nums.pop(min_index)
            operations += 1