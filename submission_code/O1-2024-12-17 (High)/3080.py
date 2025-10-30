class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        from functools import reduce
        import operator

        # First, compute the AND of the entire array
        overall_and = reduce(operator.and_, nums)
        
        # If the AND of the entire array is nonzero, the minimal sum of subarray scores
        # is that same nonzero value, and we cannot do better by splitting, so the answer is 1.
        if overall_and != 0:
            return 1
        
        # Otherwise, the overall AND is zero, so the minimal possible sum is zero.
        # We want to split the array into as many contiguous subarrays as possible,
        # each having AND = 0. A greedy approach works: we track a running AND from left to right.
        # Whenever the running AND becomes 0, we conclude one subarray and start a new one.
        
        # Since nums[i] <= 10^6, we'll use a mask that covers those bits
        ALL_ONES = (1 << 20) - 1  # covers up to 20 bits
        running_and = ALL_ONES
        count = 0
        
        for num in nums:
            running_and &= num
            if running_and == 0:
                count += 1
                running_and = ALL_ONES
        
        # count is the maximum number of subarrays we can split into
        # such that each subarray's AND = 0
        return count