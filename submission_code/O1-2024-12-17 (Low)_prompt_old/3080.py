class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        # 1) Compute the AND of the entire array.
        global_and = nums[0]
        for x in nums[1:]:
            global_and &= x
        
        # 2) If the AND of the entire array is greater than 0,
        #    the minimum possible sum is that value, achieved by
        #    taking the whole array as a single subarray. Hence result = 1.
        if global_and > 0:
            return 1
        
        # 3) Otherwise, global_and == 0. We can form subarrays each having AND = 0.
        #    To maximize the count of such subarrays, greedily form a subarray
        #    whenever its running AND becomes 0, then start a new subarray.
        count_subarrays = 0
        curr_and = (1 << 20) - 1  # a mask large enough (about 1 million)
        
        for x in nums:
            curr_and &= x
            if curr_and == 0:
                count_subarrays += 1
                curr_and = (1 << 20) - 1
        
        return count_subarrays