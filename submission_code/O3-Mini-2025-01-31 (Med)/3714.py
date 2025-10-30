from typing import List

MOD = 10**9 + 7

class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        # Explanation:
        # We need to add, for each subsequence (non‐empty) of size at most k, the sum of its minimum and maximum.
        # A useful observation is that we can “split” the sum into a sum over contributions 
        # coming from an element playing the role of maximum and one from playing the role of minimum.
        #
        # Suppose we sort the array. Then:
        #  • For each element at index i (0-indexed) acting as maximum,
        #    any subsequence that ends with element nums[i] is formed by choosing an (possibly empty)
        #    subset from indices 0..i. But because the subsequence’s total size must be at most k,
        #    the number of elements chosen from the earlier indices can be at most (k-1).
        #    Thus, the count of such subsequences is:
        #       max_count[i] = sum_{r=0}^{min(i, k-1)} C(i, r)
        #
        #  • Similarly, if nums[i] plays the role of minimum, then the subsequence is formed by picking
        #    any subset (of size at most k-1) from indices (i+1..n-1). That is, 
        #       min_count[i] = sum_{r=0}^{min(n-i-1, k-1)} C(n-i-1, r)
        #
        # Then each element nums[i] contributes
        #    nums[i] * (max_count[i] + min_count[i])
        # and the answer is the sum (modulo MOD) over all indices i.
        #
        # To efficiently compute these counts we take advantage that k <= 70.
        # We can use a dynamic programming approach that “builds” the binomial sum
        # row–by–row for i = 0,1,...,n-1.
        
        n = len(nums)
        nums.sort()
        
        # DP for maximum contributions:
        # Let dp[r] hold the current row: dp[r] = C(i, r) for r in 0..k-1.
        # For any row i (i from 0 to n-1) the count for the "maximum" contribution is:
        #      max_count[i] = sum_{r=0}^{min(i, k-1)} dp[r]
        max_count = [0] * n
        dp = [0] * k
        dp[0] = 1  # For i=0, C(0, 0)=1.
        max_count[0] = 1
        
        # Process rows i = 1 ... n-1.
        for i in range(1, n):
            # For the new row, the maximum valid r index is min(i, k-1)
            rlimit = i if i < k else k - 1
            # Update dp in reverse so that dp[r-1] is still from the previous row.
            for r in range(rlimit, 0, -1):
                dp[r] = (dp[r] + dp[r-1]) % MOD
            # dp[0] remains 1.
            s = 0
            for r in range(rlimit + 1):
                s = (s + dp[r]) % MOD
            max_count[i] = s
        
        # DP for minimum contributions:
        # We now count, for each index i in the sorted array,
        # the number of subsequences (from elements right of i) of length at most k-1.
        # We compute these counts by processing the array in reverse.
        min_count = [0] * n
        rev_dp = [0] * k
        rev_dp[0] = 1  # For the last element, there are no following elements.
        min_count[n - 1] = 1  # Only one subsequence: the element itself.
        L = 0  # Number of elements in the "suffix" that have been processed so far.
        # Process indices from n-2 downto 0.
        for i in range(n - 2, -1, -1):
            L += 1  # Now there are L = n - i - 1 elements after index i.
            rlimit = L if L < k else k - 1
            for r in range(rlimit, 0, -1):
                rev_dp[r] = (rev_dp[r] + rev_dp[r-1]) % MOD
            s = 0
            for r in range(rlimit + 1):
                s = (s + rev_dp[r]) % MOD
            min_count[i] = s
        
        # Finally, each element contributes as minimum and maximum:
        total = 0
        for i in range(n):
            total = (total + nums[i] * (max_count[i] + min_count[i])) % MOD
        
        return total

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.minMaxSums([1,2,3], 2))  # Expected output: 24
    # Example 2:
    print(sol.minMaxSums([5,0,6], 1))  # Expected output: 22
    # Example 3:
    print(sol.minMaxSums([1,1,1], 2))  # Expected output: 12