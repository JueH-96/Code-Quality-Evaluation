from typing import List

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        
        # Modular inverse of 2 modulo MOD
        # We need 2^(-1) mod (10^9 + 7). Since 10^9 + 7 is a prime number,
        # we can use Fermat's Little Theorem: a^(p-2) = a^(-1) (mod p).
        inv2 = pow(2, MOD - 2, MOD) # Calculate modular multiplicative inverse
        
        # dp[j] will store the sum of (inv2)^|t| for all subsequences t
        # using elements processed so far, such that sum(t) == j.
        # The size is k + 1 to cover sums from 0 to k.
        dp = [0] * (k + 1)
        
        # Base case: Before processing any elements, there is one subsequence: the empty sequence [].
        # Sum is 0, length is 0. The term is (inv2)^0 = 1.
        # This empty subsequence contributes to the sum of (inv2)^|t| for sum 0.
        dp[0] = 1
        
        # Iterate through each number in nums
        for num in nums:
            # Update DP table. Iterate downwards to use values from the previous step correctly.
            # For each possible target sum j (from k down to 0), we consider subsequences
            # summing to j using elements up to and including the current one.
            # These subsequences can either include 'num' or not.
            # The existing value in dp[j] (before the update in the inner loop) represents
            # the sum of (inv2)^|t| for subsequences summing to j using previous elements *without* including 'num'.
            
            # If we include 'num', the previous subsequence must have summed to (j - num).
            # We iterate j downwards from k. If j >= num, we can potentially form sum j by including 'num'.
            # Subsequences summing to (j - num) using elements *before* 'num' had a sum of (inv2)^|t'| terms equal to dp[j - num] (the old value).
            # By adding 'num', the new subsequence t = t' U {num} has sum j and length |t'| + 1.
            # Its contribution is (inv2)^(|t'| + 1) = (inv2)^|t'| * inv2.
            # The sum of contributions from all such subsequences t' (summing to j-num before) is dp[j-num] * inv2.
            # We add this new contribution to dp[j].
            
            for j in range(k, -1, -1):
                # We can only form a sum j by including 'num' if j >= num.
                # The previous sum required is j - num.
                if j >= num:
                    dp[j] = (dp[j] + dp[j - num] * inv2) % MOD
                    
        # After processing all numbers, dp[k] holds the sum of (inv2)^|t|
        # over all subsequences t of the original nums array that sum to k.
        
        # The problem asks for Sum_{s subset of nums} power(s).
        # power(s) = number of subsequences of s summing to k.
        # Sum power(s) = Sum_{s subset of nums} Sum_{t subset of s, sum(t)=k} 1
        # Change order of summation: Sum_{t subset of nums, sum(t)=k} Sum_{s subset of nums, t subset of s} 1
        # The inner sum is the number of subsequences s of nums that contain t.
        # If t is formed by |t| elements from nums, there are n - |t| elements in nums not in t.
        # Each of these n - |t| elements can either be in s or not. So there are 2^(n - |t|) such s.
        # Total sum = Sum_{t subset of nums, sum(t)=k} 2^(n - |t|).
        # Sum(2^(n - |t|)) = Sum(2^n * 2^(-|t|)) = 2^n * Sum((2^(-1))^|t|).
        
        # Calculate 2^n mod MOD
        pow2n = pow(2, len(nums), MOD)
        
        # The sum Sum((inv2)^|t|) for subsequences summing to k is stored in dp[k].
        # Final answer is (dp[k] * pow2n) mod MOD.
        result = (dp[k] * pow2n) % MOD
        
        return result