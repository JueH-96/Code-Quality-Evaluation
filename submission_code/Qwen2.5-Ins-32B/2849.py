from typing import List

class Solution:
    def sumImbalanceNumbers(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        for i in range(n):
            s = set()
            cnt = 0
            for j in range(i, n):
                s.add(nums[j])
                if nums[j] + 1 in s:
                    cnt -= 1
                if nums[j] - 1 in s:
                    cnt -= 1
                if nums[j] + 1 not in s and nums[j] - 1 not in s:
                    cnt += 1
                ans += cnt
        return ans