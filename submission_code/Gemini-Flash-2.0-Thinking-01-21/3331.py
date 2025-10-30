class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        operations = 0
        while True:
            all_greater_equal_k = True
            for num in nums:
                if num < k:
                    all_greater_equal_k = False
                    break
            if all_greater_equal_k:
                return operations
            
            min_val = min(nums)
            nums.remove(min_val)
            operations += 1