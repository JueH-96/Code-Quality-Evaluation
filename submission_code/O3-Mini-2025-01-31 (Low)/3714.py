from typing import List

MOD = 10**9 + 7

class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        # sort the array to use order statistics convenient for computing contributions as min and max
        nums.sort()
        n = len(nums)
        
        # Precompute factorials and inverse factorials for numbers 0..k (since k <=70)
        # We only need up to k because our combinations will be of the form comb(n, s) with s <= k.
        max_r = k  # we'll compute factorial for r up to k
        fact = [1]*(max_r+1)
        invfact = [1]*(max_r+1)
        for i in range(1, max_r+1):
            fact[i] = fact[i-1]*i % MOD
        
        # Fermat's little theorem for inverse modulo
        invfact[max_r] = pow(fact[max_r], MOD-2, MOD)
        for i in range(max_r, 0, -1):
            invfact[i-1] = invfact[i]*i % MOD
        
        # For a given n_val and s, compute comb(n_val, s).
        # We use the fact that for small s, comb(n_val, s) = n_val*(n_val-1)*...*(n_val-s+1)/fact[s]
        def comb(n_val, s):
            if s < 0 or s > n_val:
                return 0
            res = 1
            # Multiply numerator: n_val * (n_val-1)* ... * (n_val-s+1)
            for j in range(s):
                res = res * (n_val - j) % MOD
            res = res * invfact[s] % MOD
            return res
        
        # F(x) = sum_{s=0}^{k-1} comb(x, s)
        # We only sum s up to k-1; note comb(x, s)=0 if s > x.
        def F(x):
            total = 0
            # s will be 0 to k-1
            # When x < s, comb(x,s)=0, so we can break early.
            for s in range(k):
                if s > x:
                    break
                total = (total + comb(x, s)) % MOD
            return total

        ans = 0
        
        # For each element nums[i] in sorted order:
        # As the minimum in a subsequence:
        #   The remaining r-1 elements (for subsequence of length r, 1<=r<=k) need to be chosen from positions i+1...n-1.
        #   Total combinations for "nums[i]" being minimum is:
        #       Sum_{r=1}^{k} comb(n-i-1, r-1) = F(n-i-1), where F(x) = sum_{s=0}^{k-1} comb(x, s)
        #
        # As the maximum in a subsequence:
        #   The remaining r-1 elements must be chosen from positions 0...i-1. 
        #   Total combinations for "nums[i]" being maximum is: F(i), 
        #       because F(i) = sum_{s=0}^{k-1} comb(i, s).
        #
        # Since when subsequence is of length 1, the element appears twice (both as min and max), 
        # the calculation naturally counts the element multiplied by (1+1)=2.
        # And all subsequences are uniquely considered.
        for i in range(n):
            # count when nums[i] is minimum:
            count_min = F(n - i - 1)
            # count when nums[i] is maximum:
            count_max = F(i)
            # Total contribution from nums[i]
            contrib = nums[i] * (count_min + count_max) % MOD
            ans = (ans + contrib) % MOD
        
        return ans
                        
# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minMaxSums([1,2,3], 2))  # Expected output: 24
    print(sol.minMaxSums([5,0,6], 1))  # Expected output: 22
    print(sol.minMaxSums([1,1,1], 2))  # Expected output: 12