class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        
        # Count occurrences of each number
        from collections import Counter
        count = Counter(nums)
        
        # Handle zeros separately
        zero_count = count.get(0, 0)
        if 0 in count:
            del count[0]
        
        # dp[i] = number of ways to get sum i
        dp = [0] * (r + 1)
        dp[0] = 1  # empty subset
        
        # For each distinct number
        for num, cnt in count.items():
            # Update dp in reverse order to avoid using updated values
            for target in range(r, num - 1, -1):
                # We can use 1 to cnt copies of num
                for k in range(1, cnt + 1):
                    if target - k * num >= 0:
                        dp[target] = (dp[target] + dp[target - k * num]) % MOD
                    else:
                        break
        
        # Sum up all valid sums in range [l, r]
        result = 0
        for i in range(l, r + 1):
            result = (result + dp[i]) % MOD
        
        # Each subset can include 0 to zero_count zeros
        # This multiplies the count by (zero_count + 1)
        result = (result * (zero_count + 1)) % MOD
        
        return result