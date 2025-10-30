class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        count = {}
        total = {}
        
        for v in nums:
            new_count = 1  # Start new subsequence with just [v]
            new_total = v  # Sum of new subsequence
            
            # Extend subsequences ending with v-1
            if v-1 in count:
                new_count = (new_count + count[v-1]) % MOD
                new_total = (new_total + total[v-1] + v * count[v-1]) % MOD
            
            # Extend subsequences ending with v+1
            if v+1 in count:
                new_count = (new_count + count[v+1]) % MOD
                new_total = (new_total + total[v+1] + v * count[v+1]) % MOD
            
            # Update count and total for value v
            count[v] = (count.get(v, 0) + new_count) % MOD
            total[v] = (total.get(v, 0) + new_total) % MOD
        
        # Sum all totals
        result = 0
        for val in total.values():
            result = (result + val) % MOD
        return result