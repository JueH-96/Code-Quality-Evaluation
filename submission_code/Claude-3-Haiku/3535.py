class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        n = len(nums)
        mod = 10**9 + 7
        
        # Create a prefix sum array for the numbers
        prefix_sum = [0] * (max(nums) + 1)
        for num in nums:
            prefix_sum[num] += 1
        
        # Compute the count of monotonic pairs
        count = 0
        for i in range(1, max(nums) + 1):
            count += prefix_sum[i] * prefix_sum[max(nums) - i + 1]
        
        return count % mod