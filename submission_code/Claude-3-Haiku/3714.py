class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        n = len(nums)
        nums.sort()
        
        # Calculate the sum of the minimum elements
        min_sum = 0
        for i in range(k):
            min_sum += nums[i]
        
        # Calculate the sum of the maximum elements
        max_sum = 0
        for i in range(n-1, n-k-1, -1):
            max_sum += nums[i]
        
        return (min_sum + max_sum) % (10**9 + 7)