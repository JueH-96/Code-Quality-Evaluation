class Solution:
    def lengthOfLongestSubsequence(self, nums: List[int], target: int) -> int:
        dp = {0: 0}
        for num in nums:
            tmp = {}
            for j in list(dp.keys()):
                new_j = j + num
                if new_j > target:
                    continue
                new_length = dp[j] + 1
                if new_j in dp:
                    if new_length > dp[new_j]:
                        tmp[new_j] = new_length
                else:
                    tmp[new_j] = new_length
            # Merge tmp into dp
            for key in tmp:
                if key in dp:
                    if tmp[key] > dp[key]:
                        dp[key] = tmp[key]
                else:
                    dp[key] = tmp[key]
        if target in dp:
            return dp[target]
        else:
            return -1