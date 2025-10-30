class Solution:
    def longestSubsequence(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Initialize dp and max_len_ge arrays
        dp = [[0] * 301 for _ in range(n)]
        max_len_ge = [[0] * 301 for _ in range(n)]
        
        # Precompute max_len_ge for i=0
        max_so_far = 0
        for d in range(300, -1, -1):
            current_val = dp[0][d]
            if current_val > max_so_far:
                max_so_far = current_val
            max_len_ge[0][d] = max_so_far
        
        # Process each i from 1 to n-1
        for i in range(1, n):
            # Process each j < i
            for j in range(i):
                current_d = abs(nums[i] - nums[j])
                # Get the previous maximum length from max_len_ge
                prev_max = max_len_ge[j][current_d]
                candidate = prev_max + 1 if prev_max != 0 else 2
                
                if candidate > dp[i][current_d]:
                    dp[i][current_d] = candidate
            
            # Update max_len_ge for current i
            max_so_far = 0
            for d in range(300, -1, -1):
                current_val = dp[i][d]
                if current_val > max_so_far:
                    max_so_far = current_val
                max_len_ge[i][d] = max_so_far
        
        # Determine the result
        res = 1  # Minimum possible length is 1
        for i in range(n):
            for d in range(301):
                if dp[i][d] > res:
                    res = dp[i][d]
        return res