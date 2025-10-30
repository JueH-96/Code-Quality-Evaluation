class Solution:
    def minimumIndex(self, nums: List[int]) -> int:
        n = len(nums)
        left_freq = [0] * n
        right_freq = [0] * n
        min_index = -1

        for i in range(n):
            left_freq[i] = sum(1 for j in range(i + 1) if nums[j] == nums[i])
            right_freq[i] = sum(1 for j in range(i + 1, n) if nums[j] == nums[i])

        for i in range(n):
            if left_freq[i] * 2 > i + 1 and right_freq[i] * 2 > n - i - 1:
                min_index = i

        return min_index