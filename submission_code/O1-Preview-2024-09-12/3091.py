class Solution:
    def countSubMultisets(self, nums: List[int], l: int, r: int) -> int:
        from collections import Counter
        MOD = 10**9 + 7
        counts = Counter(nums)
        total_sum = sum(nums)
        max_sum = total_sum
        dp = [0] * (max_sum + 1)
        dp[0] = 1
        
        # Process non-zero numbers
        for num in counts:
            if num == 0:
                continue
            cnt = counts[num]
            k = 1
            items = []
            while cnt > 0:
                actual_k = min(k, cnt)
                weight = num * actual_k
                items.append(weight)
                cnt -= actual_k
                k <<= 1
            for weight in items:
                for s in range(max_sum, weight - 1, -1):
                    dp[s] = (dp[s] + dp[s - weight]) % MOD
        
        # Sum dp[s] for s in [l, r]
        result = 0
        for s in range(l, r + 1):
            if s <= max_sum:
                result = (result + dp[s]) % MOD

        # Account for zeros
        if counts.get(0, 0) > 0:
            zero_count = counts[0]
            zero_ways = pow(2, zero_count, MOD)
            # Multiply result by the number of ways to choose zeros
            result = (result * zero_ways) % MOD
            # If l <= 0 <= r, need to include the cases where sum is zero (from zeros only)
            if l <= 0 <= r:
                # Exclude the empty multiset if dp[0] was not counted
                zero_only_ways = (zero_ways - 1) % MOD
                result = (result + zero_only_ways) % MOD
        else:
            # Include dp[0] if l <= 0 <= r
            if l <= 0 <= r:
                result = (result + dp[0]) % MOD

        return result % MOD