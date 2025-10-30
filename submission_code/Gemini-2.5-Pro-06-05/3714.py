from typing import List

class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)

        nums.sort()

        # Let S(m, r) = sum_{j=0 to r} C(m, j).
        # We need to compute sums_of_combos[m] = S(m, k-1) for m = 0 to n-1.
        # We use dynamic programming with the recurrence:
        # S(m, r) = S(m-1, r) + S(m-1, r-1).
        
        sums_of_combos = [0] * n
        
        # Since k is a positive integer, k >= 1.
        # DP state: s_prev[j] stores S(i-1, j).
        # We compute S(i, j) and store it in s_curr.
        s_prev = [0] * k

        # Base case for DP: i = 0
        # S(0, j) = sum_{l=0 to j} C(0, l) = 1 for all j >= 0.
        for j in range(k):
            s_prev[j] = 1
        sums_of_combos[0] = s_prev[k-1]

        # Fill DP for i = 1 to n-1
        for i in range(1, n):
            s_curr = [0] * k
            s_curr[0] = 1  # S(i, 0) = C(i, 0) = 1
            for j in range(1, k):
                # Apply the recurrence S(i, j) = S(i-1, j) + S(i-1, j-1)
                s_curr[j] = (s_prev[j] + s_prev[j-1]) % MOD
            
            sums_of_combos[i] = s_curr[k-1]
            s_prev = s_curr

        # Calculate the final total sum
        total_sum = 0
        for i in range(n):
            num = nums[i]
            
            # Count how many times nums[i] is a maximum.
            # This happens if we pick nums[i] and j other elements (0 <= j <= k-1)
            # from the i elements before it in the sorted array.
            # The number of ways is sum_{j=0 to k-1} C(i, j) = sums_of_combos[i].
            count_max = sums_of_combos[i]
            
            # Count how many times nums[i] is a minimum.
            # This happens if we pick nums[i] and j other elements (0 <= j <= k-1)
            # from the n-1-i elements after it.
            # The number of ways is sum_{j=0 to k-1} C(n-1-i, j) = sums_of_combos[n-1-i].
            count_min = sums_of_combos[n - 1 - i]
            
            # Add this element's total contribution to the sum
            contribution = (num * (count_max + count_min)) % MOD
            total_sum = (total_sum + contribution) % MOD
            
        return total_sum