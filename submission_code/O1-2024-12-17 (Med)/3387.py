class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        # Sort the array
        nums.sort()
        
        n = len(nums)
        mid = n // 2  # Index of the median in the sorted array
        
        # We will accumulate the total cost here
        total_cost = 0
        
        # For elements before the median, ensure they are not bigger than k
        for i in range(mid):
            if nums[i] > k:
                total_cost += nums[i] - k
                
        # For the median element, make it exactly k
        total_cost += abs(nums[mid] - k)
        
        # For elements after the median, ensure they are not smaller than k
        for i in range(mid + 1, n):
            if nums[i] < k:
                total_cost += k - nums[i]
                
        return total_cost