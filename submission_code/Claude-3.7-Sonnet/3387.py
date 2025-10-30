class Solution:
    def minOperationsToMakeMedianK(self, nums: List[int], k: int) -> int:
        n = len(nums)
        median_index = n // 2
        
        # Count elements already meeting our conditions
        less_equal_k = sum(1 for num in nums if num <= k)
        greater_equal_k = sum(1 for num in nums if num >= k)
        
        # How many elements we need for each condition
        need_less_equal = median_index + 1
        need_greater_equal = n - median_index
        
        operations = 0
        
        # If we don't have enough elements ≤ k
        if less_equal_k < need_less_equal:
            # We need to reduce some elements > k to be k
            # Choose those closest to k to minimize operations
            elements_to_reduce = sorted([num for num in nums if num > k])[:need_less_equal - less_equal_k]
            operations += sum(num - k for num in elements_to_reduce)
        
        # If we don't have enough elements ≥ k
        if greater_equal_k < need_greater_equal:
            # We need to increase some elements < k to be k
            # Choose those closest to k to minimize operations
            elements_to_increase = sorted([num for num in nums if num < k], reverse=True)[:need_greater_equal - greater_equal_k]
            operations += sum(k - num for num in elements_to_increase)
        
        return operations