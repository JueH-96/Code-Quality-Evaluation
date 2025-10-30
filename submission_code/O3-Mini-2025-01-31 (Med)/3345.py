from typing import List

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        mod = 10**9 + 7
        n = len(nums)
        
        # Explanation:
        # We need to sum the "power" over all subsequences S of nums.
        # For any subsequence S ⊆ nums, its "power" is defined as the number of subsequences (T ⊆ S)
        # that have a sum equal to k.
        #
        # Instead of iterating over every S (which is exponential), we reverse the order:
        # Count every pair (T, S) where T ⊆ S ⊆ nums and sum(T)==k.
        #
        # For any fixed T such that sum(T) == k, note that any S with T ⊆ S can be formed by choosing
        # any subset from the remaining elements (N \ T). There are 2^(n - |T|) such S.
        # Thus the final answer is:
        #          sum (over T with sum==k) 2^(n - |T|)
        #
        # Our next task is to count, for each possible cardinality c, the number of subsets T of nums
        # with sum equal to k and |T| = c. We can use dynamic programming with dp[s][c] which represents
        # the count of subsets (of some prefix of nums) with sum s using exactly c elements.
        #
        # After processing all nums, the answer is:
        #   answer = sum_{c=0}^{n} ( dp[k][c] * 2^(n - c) )
        
        # Initialize dp with dp[s][c] for s = 0..k and c = 0..n.
        dp = [[0]*(n+1) for _ in range(k+1)]
        dp[0][0] = 1  # one way to have sum 0 with 0 elements.
        
        for num in nums:
            # For each new number, update dp.
            newdp = [row[:] for row in dp]  # copy current dp (this corresponds to not including this num in T)
            for s in range(k+1):
                for c in range(n):
                    if dp[s][c]:
                        ns = s + num
                        if ns <= k:
                            newdp[ns][c+1] = (newdp[ns][c+1] + dp[s][c]) % mod
            dp = newdp
        
        # Precompute powers of 2 modulo mod; pow2[i] = 2^i mod mod.
        pow2 = [1]*(n+1)
        for i in range(1, n+1):
            pow2[i] = (pow2[i-1] * 2) % mod
        
        ans = 0
        # For each possible size c, each subset T of that size with sum k contributes 2^(n - c)
        # (because S can be any superset of T in nums).
        for c in range(n+1):
            if dp[k][c]:
                ans = (ans + dp[k][c] * pow2[n-c]) % mod
        
        return ans


# The following code is for local testing.
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    print(sol.sumOfPower([1,2,3], 3))  # Expected output: 6
    # Example 2:
    print(sol.sumOfPower([2,3,3], 5))  # Expected output: 4
    # Example 3:
    print(sol.sumOfPower([1,2,3], 7))  # Expected output: 0