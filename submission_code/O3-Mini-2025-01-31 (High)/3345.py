from typing import List

MOD = 10**9 + 7

class Solution:
    def sumOfPower(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # We want to sum for every subsequence S ⊆ nums the number of its subsequences with sum equal to k.
        # Observe that a valid subsequence T ⊆ nums with sum == k will be a subsequence of many S.
        # In fact, if T has t elements then any S that contains T can be formed by choosing any subset
        # of the remaining (n-t) elements. Hence T will be counted 2^(n-t) times.
        # Thus, the answer equals: sum_{T ⊆ nums with sum==k} 2^(n - |T|).
        #
        # We now need to count, for each possible t (the size of T), how many ways are there to
        # choose a subsequence T (from all n elements) with sum exactly k and size t.
        # We can use DP with state dp[s][t] = number of ways to pick a subsequence T (from some prefix)
        # with total sum s and having chosen exactly t elements.
        
        # dp[s][t]: number of ways to obtain sum s with exactly t elements; s in [0, k], t in [0, n]
        dp = [[0] * (n + 1) for _ in range(k + 1)]
        dp[0][0] = 1  # one way: choose nothing
        
        # Process each element from nums.
        for num in nums:
            # Create a new dp array for the new state after considering current num.
            new_dp = [[0] * (n + 1) for _ in range(k + 1)]
            for s in range(k + 1):
                for t in range(n + 1):
                    if dp[s][t]:
                        # Option 1: Do not pick the current element.
                        new_dp[s][t] = (new_dp[s][t] + dp[s][t]) % MOD
                        # Option 2: Pick the current element if the new sum does not exceed k.
                        if s + num <= k and t + 1 <= n:
                            new_dp[s + num][t + 1] = (new_dp[s + num][t + 1] + dp[s][t]) % MOD
            dp = new_dp
        
        # Precompute powers of two modulo MOD.
        # For a valid subsequence T of size t, it contributes 2^(n-t) to the final answer.
        pow2 = [1] * (n + 1)
        for i in range(1, n + 1):
            pow2[i] = (pow2[i - 1] * 2) % MOD
        
        result = 0
        # Sum over all valid T (with different sizes) that sum to k.
        for t in range(n + 1):
            result = (result + dp[k][t] * pow2[n - t]) % MOD
        return result

# The following test cases illustrate the solution:
if __name__ == '__main__':
    sol = Solution()
    # Example 1:
    # For nums = [1, 2, 3] and k = 3:
    # Valid T: [1, 2] and [3].
    # Their contributions: [1,2] -> 2^(3-2)=2 and [3] -> 2^(3-1)=4, so answer = 2+4 = 6.
    print(sol.sumOfPower([1, 2, 3], 3))  # Expected output: 6
    
    # Example 2:
    # For nums = [2,3,3] and k = 5:
    # Valid T: Two choices of [2,3] (choosing one of the 3's).
    # Each contributes 2^(3-2)=2. Answer = 2+2 = 4.
    print(sol.sumOfPower([2, 3, 3], 5))  # Expected output: 4
    
    # Example 3:
    # For nums = [1,2,3] and k = 7:
    # There is no subsequence with sum 7.
    print(sol.sumOfPower([1, 2, 3], 7))  # Expected output: 0