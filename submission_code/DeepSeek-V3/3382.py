class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        from collections import defaultdict

        # To store the indices of each number
        num_to_indices = defaultdict(list)
        for idx, num in enumerate(nums):
            num_to_indices[num].append(idx)

        total = 0

        for num in num_to_indices:
            indices = num_to_indices[num]
            n = len(indices)
            # For each pair of indices (i, j), where i <= j, the subarray nums[i..j] is valid
            # The number of such pairs is n*(n+1)//2
            total += n * (n + 1) // 2

        return total