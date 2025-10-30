class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        
        # dp[i] = set of (j1, j2, or1, or2) tuples achievable using the first i elements
        # j1 = number of elements chosen for set1
        # j2 = number of elements chosen for set2
        # or1 = OR of elements in set1
        # or2 = OR of elements in set2
        dp = [set() for _ in range(n + 1)]
        dp[0].add((0, 0, 0, 0))
        
        for i in range(n):
            for j1, j2, or1, or2 in dp[i]:
                # Don't choose element i
                dp[i + 1].add((j1, j2, or1, or2))
                
                # Choose element i for set1
                if j1 < k:
                    dp[i + 1].add((j1 + 1, j2, or1 | nums[i], or2))
                
                # Choose element i for set2
                if j2 < k:
                    dp[i + 1].add((j1, j2 + 1, or1, or2 | nums[i]))
        
        max_value = 0
        for j1, j2, or1, or2 in dp[n]:
            if j1 == k and j2 == k:
                max_value = max(max_value, or1 ^ or2)
        
        return max_value