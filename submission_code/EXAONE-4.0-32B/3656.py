class Solution:
    def minimumOperations(self, nums: List[int]) -> int:
        n = len(nums)
        max_ops = (n + 2) // 3
        
        for k in range(max_ops + 1):
            start_index = 3 * k
            if start_index >= n:
                return k
            remaining = nums[start_index:]
            if len(remaining) == len(set(remaining)):
                return k