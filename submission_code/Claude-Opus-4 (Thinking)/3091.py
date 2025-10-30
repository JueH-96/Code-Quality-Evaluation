class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        
        # Count occurrences of each element
        count = Counter(nums)
        
        # Initialize DP array
        dp = [0] * (r + 1)
        dp[0] = 1  # Empty subset
        
        # Handle zeros separately - they multiply possibilities without changing sum
        if 0 in count:
            zero_count = count[0]
            for i in range(r + 1):
                dp[i] = dp[i] * (zero_count + 1) % MOD
            del count[0]
        
        # Process each non-zero unique element
        for num, cnt in count.items():
            # Update dp array for this element
            # Process in reverse order to avoid using updated values
            for i in range(r, num - 1, -1):
                temp = 0
                # Try using 1, 2, ..., min(cnt, i//num) copies of num
                for k in range(1, min(cnt, i // num) + 1):
                    temp = (temp + dp[i - k * num]) % MOD
                dp[i] = (dp[i] + temp) % MOD
        
        # Sum up the counts for sums in the range [l, r]
        result = 0
        for i in range(l, r + 1):
            result = (result + dp[i]) % MOD
        
        return result