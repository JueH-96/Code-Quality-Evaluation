from typing import List

class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        nums.sort()

        # Precompute factorials and inverse factorials modulo MOD
        # Max value for n in nCr is n (or n-1, i). Max value for r in nCr is k-1.
        # So precomputation up to n is sufficient for the base n in nCr,
        # and k-1 for the base r in nCr. Since k <= 70, k-1 is small.
        # The maximum needed factorial/inverse factorial is for n.
        MAX_PRECOMP = n
        fact = [1] * (MAX_PRECOMP + 1)
        invFact = [1] * (MAX_PRECOMP + 1)

        for i in range(2, MAX_PRECOMP + 1):
            fact[i] = (fact[i - 1] * i) % MOD

        # Modular inverse using Fermat's Little Theorem
        # invFact[i] = pow(fact[i], MOD - 2, MOD)
        # Compute from MAX_PRECOMP down to 1 using invFact[i] = invFact[i+1] * (i+1)
        # invFact[0] is 1 (since 0! = 1)
        invFact[MAX_PRECOMP] = pow(fact[MAX_PRECOMP], MOD - 2, MOD)
        for i in range(MAX_PRECOMP - 1, -1, -1): # Loop down to 0
            invFact[i] = (invFact[i + 1] * (i + 1)) % MOD


        # Function to compute combinations nCr % MOD
        def nCr_mod_p(n_comb, r_comb):
            if r_comb < 0 or r_comb > n_comb:
                return 0
            # The arguments n_comb and r_comb are always within the bounds
            # of precomputed fact/invFact arrays based on how the function is called.
            # n_comb will be n-1-i or i, both <= n-1.
            # r_comb will be p, which is at most k-1.
            
            return (fact[n_comb] * invFact[r_comb] % MOD * invFact[n_comb - r_comb] % MOD) % MOD

        # Function to compute sum of combinations C(n_sum, 0) + ... + C(n_sum, r_sum) % MOD
        # Sum C(n_sum, p) for p from 0 up to min(r_sum, n_sum)
        # This calculates the number of ways to choose *at most* r_sum elements from n_sum available elements.
        def sum_nCr_mod_p(n_sum, r_sum):
            total = 0
            # We choose p elements, where 0 <= p <= r_sum.
            # Also, we can only choose p elements if p <= n_sum.
            # So p goes from 0 up to min(r_sum, n_sum).
            upper_p = min(r_sum, n_sum)
            
            # If n_sum < 0, there are no elements to choose from, sum is 0.
            # This case should not be reached given n-1-i >= 0 and i >= 0 for 0 <= i < n.
            if n_sum < 0:
                 return 0

            # The loop goes from 0 up to upper_p inclusive.
            # If upper_p is -1 (e.g., r_sum is -1, which happens if k-1 < 0, i.e., k=0, not possible by constraint k>=1),
            # range(0) is empty, loop is skipped, total remains 0. Correct.
            for p in range(upper_p + 1):
                total = (total + nCr_mod_p(n_sum, p)) % MOD
            return total

        total_sum = 0

        # Iterate through each number in the sorted array
        # A number nums[i] contributes to the sum whenever it's the minimum or maximum
        # of a valid subsequence (length at most k).
        for i in range(n):
            num = nums[i]

            # Count subsequences where nums[i] is the minimum
            # For nums[i] to be the minimum, it must be included in the subsequence.
            # All other elements must be chosen from nums[i+1...n-1] (n-1-i elements).
            # The subsequence includes 1 element (nums[i]). Its total length is 1 + (number of chosen elements from right).
            # Total length must be at most k, so 1 + p <= k => p <= k-1.
            # We need to choose p elements from n-1-i available elements to the right, where 0 <= p <= k-1.
            # The number of ways is sum_{p=0}^{k-1} C(n-1-i, p).
            num_elements_right = n - 1 - i
            count_min = sum_nCr_mod_p(num_elements_right, k - 1)

            # Count subsequences where nums[i] is the maximum
            # For nums[i] to be the maximum, it must be included in the subsequence.
            # All other elements must be chosen from nums[0...i-1] (i elements).
            # The subsequence includes 1 element (nums[i]). Its total length is 1 + (number of chosen elements from left).
            # Total length must be at most k, so 1 + p <= k => p <= k-1.
            # We need to choose p elements from i available elements to the left, where 0 <= p <= k-1.
            # The number of ways is sum_{p=0}^{k-1} C(i, p).
            num_elements_left = i
            count_max = sum_nCr_mod_p(num_elements_left, k - 1)

            # Add contribution: num * count_min (for minimum role) + num * count_max (for maximum role)
            # Apply modulo at each addition and multiplication step to prevent overflow
            total_sum = (total_sum + num * count_min) % MOD
            total_sum = (total_sum + num * count_max) % MOD

        return total_sum