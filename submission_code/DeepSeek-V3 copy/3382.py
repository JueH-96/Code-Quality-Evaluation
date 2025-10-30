class Solution:
    def numberOfSubarrays(self, nums: List[int]) -> int:
        from collections import defaultdict

        # Create a dictionary to store the indices of each number
        num_indices = defaultdict(list)
        for idx, num in enumerate(nums):
            num_indices[num].append(idx)

        total = 0

        # Iterate through each unique number in the list
        for num in num_indices:
            indices = num_indices[num]
            n = len(indices)
            # For each pair of indices (i, j), where i <= j, the subarray from i to j is valid
            # The number of such pairs is n*(n+1)//2
            total += n * (n + 1) // 2

        return total