from typing import List

class Solution:
    def minMaxSums(self, nums: List[int], k: int) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        
        # Sort the array
        nums.sort()
        
        # Helper function for modular exponentiation
        def pow_mod(base, exp):
            return pow(base, exp, MOD)

        # Helper function for modular inverse
        def inv_mod(n_val):
            # Assumes MOD is prime and n_val > 0
            return pow_mod(n_val, MOD - 2)

        # Helper function to compute sum(C(n_val, j) for j=0 to k_limit) mod MOD
        # C(n_val, j) is number of ways to choose j items from n_val items.
        # We need sum(C(n_val, j) for j=0, ..., k_limit)
        def get_sum_C(n_val, k_limit):
            # Sum C(n_val, j) for j = 0, 1, ..., k_limit.
            # C(n_val, j) is 0 if j < 0 or j > n_val.
            # So the sum is actually for j = 0, 1, ..., min(k_limit, n_val).

            if k_limit < 0 or n_val < 0:
                return 0 # Cannot choose k_limit elements if k_limit < 0. Cannot choose from n_val if n_val < 0.

            upper_j = min(k_limit, n_val)

            current_C = 1 # C(n_val, 0)
            total_sum = 0 # We will add C(n_val, j) for j from 0 to upper_j

            # Loop j from 0 up to upper_j
            for j in range(upper_j + 1):
                if j == 0:
                    # C(n_val, 0) = 1
                    current_C = 1
                else:
                    # Calculate C(n_val, j) from C(n_val, j-1)
                    # C(N, J) = C(N, J-1) * (N - J + 1) / J
                    # current_C is C(n_val, j-1)
                    term = (current_C * (n_val - j + 1)) % MOD
                    # modular inverse is needed for division by j
                    term = (term * inv_mod(j)) % MOD # inv_mod needs j > 0, which is true for j >= 1
                    current_C = term

                # Add the current combination value C(n_val, j) to the total sum
                total_sum = (total_sum + current_C) % MOD

            return total_sum

        total_overall_sum = 0

        # Iterate through the sorted array
        for i in range(n):
            # For nums[i] to be the maximum in a subsequence of length <= k,
            # it must be included, and all other elements (j of them, 0 <= j <= k-1)
            # must be chosen from nums[0...i-1] (which has i elements).
            # The number of ways to choose j elements from i is C(i, j).
            # Total count is sum(C(i, j) for j=0 to k-1).
            count_as_max = get_sum_C(i, k - 1)

            # For nums[i] to be the minimum in a subsequence of length <= k,
            # it must be included, and all other elements (j of them, 0 <= j <= k-1)
            # must be chosen from nums[i+1...n-1] (which has n-1-i elements).
            # The number of ways to choose j elements from n-1-i is C(n-1-i, j).
            # Total count is sum(C(n-1-i, j) for j=0 to k-1).
            count_as_min = get_sum_C(n - 1 - i, k - 1)

            # The contribution of nums[i] to the total sum is nums[i] * (count_as_max + count_as_min)
            # sum(max(sub)) = sum_i(nums[i] * count_as_max_for_i)
            # sum(min(sub)) = sum_i(nums[i] * count_as_min_for_i)
            # Total sum = sum(max) + sum(min) = sum_i(nums[i] * (count_as_max_for_i + count_as_min_for_i))
            
            # Sum of counts: how many times nums[i] is *a* maximum + how many times nums[i] is *a* minimum
            total_counts_for_i = (count_as_max + count_as_min) % MOD
            
            # Contribution of nums[i] to the total sum
            contribution = (nums[i] * total_counts_for_i) % MOD

            total_overall_sum = (total_overall_sum + contribution) % MOD

        return total_overall_sum