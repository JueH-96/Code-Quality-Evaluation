class Solution:
    def minimumSum(self, n: int, k: int) -> int:
        
        # To get the minimum sum, we greedily pick the smallest distinct positive integers.
        # We can pick `i` as long as `k - i` has not been picked yet.
        # This logic leads to picking from two distinct groups of numbers:
        # 1. The set {1, 2, ..., floor(k/2)}
        # 2. The set {k, k+1, k+2, ...}
        # Any number between these two groups, i.e., in the range [floor(k/2) + 1, k-1],
        # cannot be picked because its complement `k-i` would have already been
        # selected from the first group.

        # Let m be the largest number in the first group of safe-to-pick numbers.
        m = k // 2

        # Case 1: If n is small enough (n <= m), we only need to pick from the first group.
        # The elements will be 1, 2, ..., n.
        if n <= m:
            # The sum of the first n integers is n * (n + 1) / 2.
            return n * (n + 1) // 2
        
        # Case 2: If n is larger than m, we must take all numbers from the first group
        # and the remaining numbers from the start of the second group.
        else:
            # Sum of the first group: 1 + 2 + ... + m
            sum_first_part = m * (m + 1) // 2

            # We need to pick `n - m` more numbers.
            remaining_count = n - m
            
            # These numbers will be the first `remaining_count` numbers from the second group:
            # k, k+1, ..., k + remaining_count - 1.
            # We calculate the sum of this arithmetic progression.
            first_num_second_part = k
            last_num_second_part = k + remaining_count - 1
            sum_second_part = remaining_count * (first_num_second_part + last_num_second_part) // 2
            
            return sum_first_part + sum_second_part