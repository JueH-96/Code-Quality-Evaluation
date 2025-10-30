class Solution:
    def minimumSubarrayLength(self, nums: List[int], k: int) -> int:
        n = len(nums)
        min_len = float('inf')
        
        for i in range(n):
            bitwise_or = 0
            for j in range(i, n):
                bitwise_or |= nums[j]
                if bitwise_or >= k:
                    min_len = min(min_len, j - i + 1)
                    break  # Once condition is met, no need to extend this subarray
        
        return min_len if min_len != float('inf') else -1