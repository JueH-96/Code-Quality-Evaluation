class Solution:
    def maximumTripletValue(self, nums: List[int]) -> int:
        n = len(nums)
        max_val = float('-inf')
        
        # Explore all triplets (i, j, k) with i < j < k
        for i in range(n - 2):
            for j in range(i + 1, n - 1):
                for k in range(j + 1, n):
                    value = (nums[i] - nums[j]) * nums[k]
                    max_val = max(max_val, value)
        
        # If the maximum value is negative, return 0; otherwise return the maximum value.
        return max(max_val, 0)