class Solution:
    def maxSubarrays(self, nums: List[int]) -> int:
        from functools import reduce
        import operator
        
        # 1) Compute the bitwise AND of the entire array.
        global_and = reduce(operator.and_, nums)
        
        # 2) If this AND is nonzero, the minimal possible score is global_and,
        #    and using more subarrays would only increase the total AND sum.
        #    Hence, in that case, we must take the whole array as one subarray.
        if global_and != 0:
            return 1
        
        # 3) Otherwise, global_and == 0. The minimal score is 0.
        #    We then want to form as many subarrays as possible, each of whose
        #    bitwise AND is 0. Greedily cut a subarray as soon as its AND becomes 0.
        
        count = 0
        # Since nums[i] <= 10^6 (< 2^20), a safe "all-bits-set" mask up to 2^20.
        ALL_BITS_SET = (1 << 20) - 1
        
        running_and = ALL_BITS_SET
        for x in nums:
            running_and &= x
            if running_and == 0:
                count += 1
                running_and = ALL_BITS_SET
        
        return count