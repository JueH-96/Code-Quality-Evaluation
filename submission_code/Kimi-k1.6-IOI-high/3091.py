class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        MOD = 10**9 + 7
        cnt_zero = nums.count(0)
        freq = {}
        for num in nums:
            if num != 0:
                if num in freq:
                    freq[num] += 1
                else:
                    freq[num] = 1
        
        items = []
        sum_non_zero_filtered = 0
        for x in freq:
            if x <= r:
                items.append((x, freq[x]))
                sum_non_zero_filtered += x * freq[x]
        
        if sum_non_zero_filtered < l:
            return 0
        
        upper = min(r, sum_non_zero_filtered)
        dp = [0] * (upper + 1)
        dp[0] = 1
        
        for x, freq in items:
            if x > upper:
                continue
            # Unbounded knapsack step
            for s in range(x, upper + 1):
                dp[s] = (dp[s] + dp[s - x]) % MOD
            # Subtraction step
            required = (freq + 1) * x
            if required <= upper:
                for s in range(upper, required - 1, -1):
                    dp[s] = (dp[s] - dp[s - required]) % MOD
        
        total = 0
        for s in range(l, upper + 1):
            total = (total + dp[s]) % MOD
        total = total * (cnt_zero + 1) % MOD
        
        return total