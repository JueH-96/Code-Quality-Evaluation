class Solution:
    def numberOfGoodSubarraySplits(self, nums: List[int]) -> int:
        first = -1
        last = -1
        for i in range(len(nums)):
            if nums[i] == 1:
                if first == -1:
                    first = i
                last = i
        
        if first == -1:
            return 0
        
        ans = 1
        mod = 10**9 + 7
        
        count = 0
        for i in range(first + 1, last + 1):
            if nums[i] == 1:
                ans = (ans * (count + 1)) % mod
                count = 0
            else:
                count += 1
        
        return ans