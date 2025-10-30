class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        if n <= 1:
            return n
        max_len = 1
        for i in range(n):
            max_len = max(max_len, 1) # consider subsequence of length 1
        for i in range(n):
            for j in range(i + 1, n):
                current_subsequence = [nums[i], nums[j]]
                current_diff = abs(nums[j] - nums[i])
                current_len = 2
                last_index = j
                for k in range(j + 1, n):
                    next_diff = abs(nums[k] - nums[last_index])
                    if next_diff <= current_diff:
                        current_subsequence.append(nums[k])
                        current_diff = next_diff
                        current_len += 1
                        last_index = k
                max_len = max(max_len, current_len)
        return max_len