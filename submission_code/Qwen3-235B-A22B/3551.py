class Solution:
    def maximumSubarrayXor(self, nums: List[int], queries: List[List[int]]) -> List[int]:
        n = len(nums)
        if n == 0:
            return []
        
        # Initialize dp table
        dp = [[0] * n for _ in range(n)]
        
        # Base case: mask 0 (subarray length 1)
        for L in range(n):
            dp[L][0] = nums[L]
        
        # Fill dp for masks >= 1
        for mask in range(1, n):
            b = mask.bit_length() - 1
            mask_without_b = mask ^ (1 << b)
            for L in range(n - mask):
                dp[L][mask] = dp[L][mask_without_b] ^ dp[L + (1 << b)][mask_without_b]
        
        # Precompute max_mask: max_mask[L][mask] is the maximum dp[L][0..mask]
        max_mask = [[0] * n for _ in range(n)]
        for L in range(n):
            current_max = -float('inf')
            for mask in range(n - L):
                if dp[L][mask] > current_max:
                    current_max = dp[L][mask]
                max_mask[L][mask] = current_max
        
        # Process each query
        res = []
        for l, r in queries:
            max_val = -float('inf')
            for L in range(l, r + 1):
                mask_max = r - L
                if mask_max < 0 or mask_max >= n - L:
                    continue  # mask_max can't exceed the available masks for L
                current = max_mask[L][mask_max]
                if current > max_val:
                    max_val = current
            res.append(max_val)
        
        return res