class Solution:
    def minimumSeconds(self, nums):
        n = len(nums)
        max_val = max(nums)
        max_indices = [i for i, x in enumerate(nums) if x == max_val]
        min_dist = float('inf')
        for i in range(n):
            min_dist = min(min_dist, min(abs(i - j), n - abs(i - j)) for j in max_indices)
        return min_dist