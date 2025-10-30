import math
from typing import List

class Solution:
    def numberOfSubsequences(self, nums: List[int]) -> int:
        n = len(nums)
        max_val = 1000 # Constraint 1 <= nums[i] <= 1000

        # Precompute suffix counts
        # sc[i][v] = count of value v in nums[i...n-1]
        suffix_counts = [[0] * (max_val + 1) for _ in range(n + 1)]
        # Iterate backwards to compute suffix counts
        for i in range(n - 1, -1, -1):
            # Copy counts from the next index
            for v in range(1, max_val + 1):
                suffix_counts[i][v] = suffix_counts[i + 1][v]
            # Add count for the current element
            # Constraints ensure nums[i] is within [1, max_val]
            suffix_counts[i][nums[i]] += 1

        total_count = 0
        
        # lc[v] = count of value v in nums[0...q-2]
        left_counts = [0] * (max_val + 1)

        # Iterate through q and r
        # A special subsequence is (p, q, r, s) with p < q < r < s and q-p > 1, r-q > 1, s-r > 1.
        # This means:
        # p <= q - 2
        # q <= r - 2
        # r <= s - 2
        # From these, we get p <= q-2 <= r-4 <= s-6.
        # Minimum indices: p=0, q=2, r=4, s=6. Requires n >= 7.
        # Maximum indices: s=n-1, r=n-3, q=n-5, p=n-7. Requires n >= 7.

        # Iterate q from its minimum possible value (2) to its maximum possible value (n-5)
        for q in range(2, n - 4): # q iterates from 2 up to n-5 (inclusive)
            # Add nums[q-2] to left_counts for the current q
            # nums[0...q-3] counts are already in left_counts from the previous q iteration.
            # Adding nums[q-2] updates left_counts to cover nums[0...q-2].
            # The index q-2 ranges from 0 (when q=2) to n-6 (when q=n-5).
            left_counts[nums[q-2]] += 1

            # Iterate r from its minimum possible value (q+2) to its maximum possible value (n-3)
            for r in range(q + 2, n - 2): # r iterates from q+2 up to n-3 (inclusive)
                # We have fixed q and r.
                # The valid indices for p are in [0, q-2]. Counts are maintained in left_counts.
                # The valid indices for s are in [r+2, n-1]. Counts are available in suffix_counts[r+2].

                val_q = nums[q]
                val_r = nums[r]
                
                # Values are guaranteed to be >= 1 from constraints
                common_divisor = math.gcd(val_q, val_r)
                q_prime = val_q // common_divisor
                r_prime = val_r // common_divisor

                # The condition nums[p] * nums[r] == nums[q] * nums[s] can be written as
                # nums[p] / nums[q] == nums[s] / nums[r].
                # Using q' = nums[q]/g and r' = nums[r]/g, this is nums[p] / (g * q') == nums[s] / (g * r').
                # nums[p] * r' == nums[s] * q'.
                # Since gcd(q', r') = 1, q' must divide nums[p] and r' must divide nums[s].
                # Let nums[p] = m * q' and nums[s] = m * r' for some positive integer m.

                # We need to find the range of m.
                # 1 <= nums[p] <= 1000 => 1 <= m * q_prime <= 1000 => m <= 1000 // q_prime
                # 1 <= nums[s] <= 1000 => 1 <= m * r_prime <= 1000 => m <= 1000 // r_prime
                # m must be >= 1 as nums are positive.
                
                max_m = min(1000 // q_prime, 1000 // r_prime)

                # Iterate through possible values of m
                for m in range(1, max_m + 1):
                    val_p = m * q_prime
                    val_s = m * r_prime

                    # val_p and val_s are guaranteed to be in [1, 1000] by the max_m calculation
                    # val_p >= 1 and val_s >= 1 because m >= 1 and q_prime, r_prime >= 1

                    count_p = left_counts[val_p]
                    
                    # The starting index for s is r+2.
                    # r iterates up to n-3, so r+2 iterates up to n-1.
                    # suffix_counts[r+2] is a valid access because r+2 <= n-1+2 = n.
                    # The index range for suffix_counts is [0, n].
                    count_s = suffix_counts[r + 2][val_s]

                    total_count += count_p * count_s

        return total_count