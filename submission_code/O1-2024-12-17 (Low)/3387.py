class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        # Sort the array so that we can control positions w.r.t. the median index
        nums.sort()
        n = len(nums)
        mid = n // 2  # Median index (0-based) as per problem statement (larger of two for even n)
        
        # Cost to ensure all elements before mid are <= k
        # We only pay if nums[i] > k (we reduce it to k)
        cost_left = 0
        for i in range(mid):
            if nums[i] > k:
                cost_left += nums[i] - k
        
        # Cost to transform the median element nums[mid] into k
        cost_median = abs(nums[mid] - k)
        
        # Cost to ensure all elements after mid are >= k
        # We only pay if nums[i] < k (we increase it to k)
        cost_right = 0
        for i in range(mid + 1, n):
            if nums[i] < k:
                cost_right += k - nums[i]
        
        return cost_left + cost_median + cost_right