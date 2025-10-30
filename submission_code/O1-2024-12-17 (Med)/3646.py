class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Maximum possible value in nums is 10^5
        max_val = 100000
        
        # count[v] will hold the number of good subsequences ending with value v
        # subSum[v] will hold the sum of sums of those subsequences
        count = [0] * (max_val + 1)
        subSum = [0] * (max_val + 1)
        
        total_sum = 0
        
        for x in nums:
            # Every new element x can:
            #   1) start a new subsequence by itself (count = 1, sum = x)
            #   2) extend subsequences ending with x-1
            #   3) extend subsequences ending with x+1
            c = 1
            s = x
            
            if x > 0:
                c += count[x - 1]
                s += subSum[x - 1] + x * count[x - 1]
            
            if x < max_val:
                c += count[x + 1]
                s += subSum[x + 1] + x * count[x + 1]
            
            c %= MOD
            s %= MOD
            
            # The newly formed subsequences contribute 's' to the global total
            total_sum = (total_sum + s) % MOD
            
            # Update dp arrays for value x
            count[x] = (count[x] + c) % MOD
            subSum[x] = (subSum[x] + s) % MOD
        
        return total_sum % MOD