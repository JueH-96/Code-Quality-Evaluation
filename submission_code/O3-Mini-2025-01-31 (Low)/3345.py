from typing import List
MOD = 10**9 + 7

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # We want to compute the sum of "power" over all subsequences S ⊆ nums.
        # Here the "power" of a subsequence S is defined as the number of its subsequences T (T ⊆ S, T nonempty)
        # that have a sum equal to k.
        # We can reverse the summation: For every (nonempty) T ⊆ nums with sum == k,
        # T contributes to the power of every S that contains T.
        # The number of subsequences S with T ⊆ S ⊆ nums is exactly 2^(n - |T|) (since
        # every element not in T can either be present or absent independently in S).
        #
        # Hence, the final answer is:
        #    sum_{T ⊆ nums, T nonempty, sum(T)==k} 2^(n - |T|)
        #
        # Notice: Even if an element in nums is greater than k, it cannot be part of T,
        # so our count of valid T's using a standard subset sum DP is valid.
        #
        # We will do a DP that computes for each possible sum "s" (0 <= s <= k) and
        # each possible count "c" (number of elements in the chosen subset T) the number
        # of ways to choose T from nums with that sum and count. Then the answer is:
        #    sum_{c=0}^{n} (dp[k][c] * 2^(n - c))
        
        # Initialize dp[s][c]: ways to pick a subset T with sum = s and |T| = c.
        # dp[0][0] = 1 (choose nothing)
        dp = [[0] * (n+1) for _ in range(k+1)]
        dp[0][0] = 1
        
        # Process each number in nums
        for num in nums:
            # We update dp in a reverse manner so that the number is used at most once.
            newdp = [row[:] for row in dp]
            for s in range(k+1):
                for c in range(n):
                    if dp[s][c]:
                        ns = s + num
                        if ns <= k:
                            newdp[ns][c+1] = (newdp[ns][c+1] + dp[s][c]) % MOD
            dp = newdp

        # Precompute powers of 2: pow2[x] = 2^x mod MOD, for x = 0..n.
        pow2 = [1] * (n+1)
        for i in range(1, n+1):
            pow2[i] = (pow2[i-1] * 2) % MOD
        
        ans = 0
        # Sum over all possible sizes c of T that yield sum exactly k.
        for c in range(n+1):
            # Each valid T of size c is contained in 2^(n-c) subsequences S of nums.
            ans = (ans + dp[k][c] * pow2[n - c]) % MOD
            
        return ans

# Example usage:
if __name__ == "__main__":
    solution = Solution()
    # Example 1:
    print(solution.sumOfPower([1,2,3], 3))  # Expected output 6
    # Example 2:
    print(solution.sumOfPower([2,3,3], 5))  # Expected output 4
    # Example 3:
    print(solution.sumOfPower([1,2,3], 7))  # Expected output 0