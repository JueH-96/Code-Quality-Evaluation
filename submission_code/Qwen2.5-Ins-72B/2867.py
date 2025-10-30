class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        n = len(nums)
        last_one = -1
        count = 1
        found_one = False
        
        for i in range(n):
            if nums[i] == 1:
                if found_one:
                    count = (count * (i - last_one)) % mod
                else:
                    found_one = True
                last_one = i
        
        return count if found_one else 0