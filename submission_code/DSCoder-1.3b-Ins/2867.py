class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        good_subarrays = 0
        split_count = 0
        for num in nums:
            if num == 1:
                split_count += 1
            else:
                good_subarrays += split_count
                split_count = 0
        good_subarrays += split_count
        if good_subarrays == 0:
            return 0
        else:
            return pow(good_subarrays, mod-2, mod)