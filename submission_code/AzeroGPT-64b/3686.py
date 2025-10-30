from collections import defaultdict

class Solution:
    def beautifulSplits(self, nums: List[int]) -> int:
        
        prefixCounter = defaultdict(int)
        suffixCounter = defaultdict(int)
        ans = 0
        n = len(nums)

        if n <= 2:
            return 0

        for num in nums:
            suffixCounter[num] += 1

        for i in range(n - 1):
            prefixCounter[nums[i]] += 1
            suffixCounter[nums[i]] -= 1

            for num in nums:
                if suffixCounter[num] > 0 and (num in prefixCounter or list(prefixCounter.values()).count(1) > 1):
                    ans += 1
                    break
        
        return ans