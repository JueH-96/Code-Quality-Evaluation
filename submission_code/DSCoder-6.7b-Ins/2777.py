class Solution:
    def distinctDifferenceArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        prefix_counts = [0] * n
        suffix_counts = [0] * n
        num_counts = [0] * 51

        for num in nums:
            num_counts[num] += 1

        for i in range(n):
            if i > 0:
                prefix_counts[i] = prefix_counts[i - 1]
            if num_counts[nums[i]] == 1:
                prefix_counts[i] += 1
            num_counts[nums[i]] -= 1

        num_counts = [0] * 51
        for i in range(n - 1, -1, -1):
            if i < n - 1:
                suffix_counts[i] = suffix_counts[i + 1]
            if num_counts[nums[i]] == 0:
                suffix_counts[i] += 1
            num_counts[nums[i]] += 1

        return [prefix_counts[i] - suffix_counts[i] for i in range(n)]