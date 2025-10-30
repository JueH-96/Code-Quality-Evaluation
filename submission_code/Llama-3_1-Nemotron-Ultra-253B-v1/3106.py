from typing import List

class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        dp = {0: 0}
        for num in nums:
            temp = {}
            for s in list(dp.keys()):
                new_sum = s + num
                if new_sum > target:
                    continue
                new_length = dp[s] + 1
                if new_sum in temp:
                    if new_length > temp[new_sum]:
                        temp[new_sum] = new_length
                else:
                    temp[new_sum] = new_length
            for s in temp:
                if s in dp:
                    if temp[s] > dp[s]:
                        dp[s] = temp[s]
                else:
                    dp[s] = temp[s]
        return dp.get(target, -1)