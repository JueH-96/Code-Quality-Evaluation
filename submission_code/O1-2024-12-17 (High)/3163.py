class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            freq = {}
            distinct = 0
            for j in range(i, n):
                val = nums[j]
                freq[val] = freq.get(val, 0) + 1
                if freq[val] == 1:
                    distinct += 1
                ans += distinct * distinct
        return ans