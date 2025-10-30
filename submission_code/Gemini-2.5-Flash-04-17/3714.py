from typing import List

class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        nums.sort()

        # Precompute modular inverses up to k-1
        # We need inv[p] for p from 1 to m, where m <= k-1.
        max_inv_p = k - 1
        inv = [1] * (max_inv_p + 1)
        if max_inv_p >= 1:
            # inv[p] = pow(p, MOD - 2, MOD) using Fermat's Little Theorem
            # The base case inv[1] = pow(1, MOD - 2, MOD) = 1 is covered by initialization
            for p in range(2, max_inv_p + 1):
                 inv[p] = pow(p, MOD - 2, MOD)

        # Function to compute sum_{p=0}^{m} C(N, p) mod MOD
        # Uses C(N, p) = C(N, p-1) * (N - p + 1) / p
        def sum_combinations_up_to(N, m):
            # Handles edge cases N < 0 or m < 0
            if m < 0 or N < 0:
                return 0
            
            # If m >= N, sum is C(N, 0) + ... + C(N, N) = 2^N
            # All combinations C(N, p) for p > N are 0.
            # Sum C(N, p) for p from 0 to N is 2^N.
            if m >= N:
                return pow(2, N, MOD)
            
            # If m < N, sum is C(N, 0) + ... + C(N, m)
            current_sum = 0
            C_N_p = 1 # C(N, 0)
            current_sum = (current_sum + C_N_p) % MOD

            # Iterate p from 1 to m to compute C(N, p) and add to sum
            # We need inv[p] for p from 1 to m.
            # Since m <= k-1 = max_inv_p, our precomputed inv array is sufficient.
            for p in range(1, m + 1):
                # Calculate C(N, p) from C(N, p-1) using the recurrence:
                # C(N, p) = C(N, p-1) * (N - p + 1) / p
                # Modulo arithmetic requires modular inverse for division:
                # C(N, p) = C(N, p-1) * (N - p + 1) * inv[p] mod MOD
                
                # (N - p + 1) can be large, take modulo.
                # Since p <= m < N, N - p + 1 >= N - (N - 1) + 1 = 2 > 0. So (N - p + 1) % MOD is correct.
                term_numerator = (N - p + 1) % MOD
                C_N_p = (C_N_p * term_numerator) % MOD
                C_N_p = (C_N_p * inv[p]) % MOD
                
                current_sum = (current_sum + C_N_p) % MOD

            return current_sum

        total_sum = 0

        # Iterate through each element in the sorted array
        for i in range(n):
            # Calculate the number of subsequences where nums[i] is the minimum.
            # Such a subsequence must contain nums[i] and other elements chosen from nums[i+1...n-1].
            # There are (n - 1 - i) elements available after index i.
            # Let p be the number of additional elements chosen from these (n - 1 - i) elements.
            # The total length of the subsequence is 1 (for nums[i]) + p.
            # We require the length to be at most k: 1 + p <= k => p <= k - 1.
            # The number of ways to choose p elements from N = (n - 1 - i) is C(N, p).
            # The total count for nums[i] as minimum is the sum of C(n - 1 - i, p) for valid p.
            # Valid p are 0 <= p <= (n - 1 - i) and 0 <= p <= k - 1.
            # So p must be in the range 0 <= p <= min(n - 1 - i, k - 1).
            N_min = n - 1 - i
            m_min = min(N_min, k - 1)
            count_min = sum_combinations_up_to(N_min, m_min)

            # Calculate the number of subsequences where nums[i] is the maximum.
            # Such a subsequence must contain nums[i] and other elements chosen from nums[0...i-1].
            # There are i elements available before index i.
            # Let p be the number of additional elements chosen from these i elements.
            # The total length is 1 + p. We need 1 + p <= k => p <= k - 1.
            # The number of ways to choose p elements from N = i is C(N, p).
            # The total count for nums[i] as maximum is the sum of C(i, p) for valid p.
            # Valid p are 0 <= p <= i and 0 <= p <= k - 1.
            # So p must be in the range 0 <= p <= min(i, k - 1).
            N_max = i
            m_max = min(N_max, k - 1)
            count_max = sum_combinations_up_to(N_max, m_max)

            # nums[i] contributes as the minimum in 'count_min' subsequences, adding nums[i] to the sum 'count_min' times.
            # nums[i] contributes as the maximum in 'count_max' subsequences, adding nums[i] to the sum 'count_max' times.
            # The total contribution of nums[i] is nums[i] * count_min + nums[i] * count_max = nums[i] * (count_min + count_max).
            coeff = (count_min + count_max) % MOD
            term = (nums[i] * coeff) % MOD
            total_sum = (total_sum + term) % MOD

        return total_sum