class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 1_000_000_007

        # To obtain the minimum sum, we greedily pick the smallest available positive integers.
        # The strategy is to pick i=1, 2, 3... as long as for any number x already picked,
        # i + x != target. This is equivalent to i not being equal to target - x.
        
        # Let m = target // 2. The numbers 1, 2, ..., m can always be picked.
        # For any i in [1, m], target - i > m, so target-i cannot have been picked yet.
        m = target // 2

        # Case 1: n <= m.
        # We can pick the first n positive integers: {1, 2, ..., n}.
        # For any two numbers i, j in this set, i + j < target.
        # The sum is the sum of the first n integers.
        if n <= m:
            # Sum of 1 to n = n * (n + 1) / 2
            # Python's arbitrary-precision integers handle large numbers, so we
            # can calculate the full sum and then take the modulo.
            total_sum = (n * (n + 1)) // 2
            return total_sum % MOD

        # Case 2: n > m.
        # We first pick {1, 2, ..., m}.
        # The next available number to pick is `target`.
        # The remaining (n - m) numbers will be {target, target + 1, ...}.
        
        # The final beautiful array is {1, ..., m} U {target, ..., target + n - m - 1}.
        
        # Sum of the first part: {1, 2, ..., m}
        sum1 = (m * (m + 1)) // 2
        
        # Sum of the second part: an arithmetic progression of (n - m) terms
        # starting from `target`.
        count_rem = n - m
        start_num = target
        end_num = target + count_rem - 1
        
        # Sum of arithmetic series: count * (first + last) / 2
        sum2 = (count_rem * (start_num + end_num)) // 2
        
        total_sum = sum1 + sum2
        
        return total_sum % MOD