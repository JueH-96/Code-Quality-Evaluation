MOD = 10**9 + 7

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        n = len(nums)
        max_count = n
        max_sum = k
        
        # Precompute powers of 2 mod MOD
        pow2 = [1] * (n + 1)
        for i in range(1, n + 1):
            pow2[i] = (pow2[i - 1] * 2) % MOD
        
        # Initialize DP table
        dp = [[0] * (n + 1) for _ in range(max_sum + 1)]
        dp[0][0] = 1
        
        for num in nums:
            # Make a temporary copy of the current DP state
            tmp = [row.copy() for row in dp]
            for s in range(max_sum + 1):
                for c in range(max_count + 1):
                    if tmp[s][c] == 0:
                        continue
                    new_s = s + num
                    new_c = c + 1
                    if new_s <= max_sum and new_c <= max_count:
                        dp[new_s][new_c] = (dp[new_s][new_c] + tmp[s][c]) % MOD
        
        # Calculate the result
        result = 0
        for c in range(n + 1):
            if dp[k][c]:
                result = (result + dp[k][c] * pow2[n - c]) % MOD
        
        return result