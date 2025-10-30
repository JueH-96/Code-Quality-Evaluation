from typing import List

class Solution:
    def specialPerm(self, nums: List[int]) -> int:
        mod = 10**9 + 7
        n = len(nums)
        max_mask = 1 << n
        dp = [[0] * n for _ in range(max_mask)]
        
        # Initialize dp for masks with exactly one element
        for i in range(n):
            dp[1 << i][i] = 1
        
        # Generate all masks sorted by the number of bits set (increasing order)
        masks = []
        for mask in range(1, max_mask):
            masks.append(mask)
        masks.sort(key=lambda x: bin(x).count('1'))
        
        for mask in masks:
            # For each mask, process all possible last elements in the permutation
            for last in range(n):
                if not (mask & (1 << last)):
                    continue  # Skip if last is not in the current mask
                # Try to add each possible next element not in mask
                for curr in range(n):
                    if mask & (1 << curr):
                        continue  # Skip if curr is already in mask
                    # Check if nums[last] and nums[curr] can be adjacent
                    if nums[last] % nums[curr] == 0 or nums[curr] % nums[last] == 0:
                        new_mask = mask | (1 << curr)
                        dp[new_mask][curr] = (dp[new_mask][curr] + dp[mask][last]) % mod
        
        full_mask = (1 << n) - 1
        return sum(dp[full_mask][last] for last in range(n)) % mod