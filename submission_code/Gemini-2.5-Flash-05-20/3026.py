class Solution:
    def minimumPossibleSum(self, n: int, target: int) -> int:
        MOD = 10**9 + 7

        # k_val represents the count of numbers that are strictly less than target / 2.
        # These are the numbers 1, 2, ..., k_val.
        # Picking these numbers is always optimal as they are smaller than their
        # 'complement' (target - x) which would complete a sum to 'target'.
        k_val = (target - 1) // 2

        current_sum = 0
        numbers_picked_count = 0

        # Case 1: We need 'n' numbers, and 'n' is less than or equal to k_val.
        # This means we can pick the smallest 'n' positive integers (1, 2, ..., n).
        # In this scenario, all picked numbers 'x' are < target/2, so no two distinct
        # numbers in the array can sum to 'target' (as their sum would be < target).
        if n <= k_val:
            # Sum of an arithmetic progression: n * (n + 1) / 2
            current_sum = n * (n + 1) // 2
            numbers_picked_count = n
        else:
            # Case 2: We need more than k_val numbers.
            # We must pick all numbers from 1 to k_val, as they are the smallest and
            # picking them is always beneficial for minimizing the sum.
            
            # Sum of numbers from 1 to k_val
            current_sum = k_val * (k_val + 1) // 2
            numbers_picked_count = k_val

            # Calculate how many more numbers we need to pick.
            remaining_n = n - numbers_picked_count

            # Determine the starting value for these remaining numbers.
            # The numbers 1, ..., k_val have been picked. This means their complements
            # (target-1, target-2, ..., target-k_val) are forbidden.
            # We need to find the smallest positive integer that is neither picked nor forbidden.
            start_val = 0
            if target % 2 == 1:
                # If target is odd (e.g., target = 3, k_val = 1). Picked = {1}. Forbidden = {2}.
                # The smallest available is 3 (which is target).
                start_val = target
            else: # target % 2 == 0
                # If target is even (e.g., target = 6, k_val = 2). Picked = {1, 2}. Forbidden = {4, 5}.
                # The smallest available is 3 (which is target // 2).
                start_val = target // 2
            
            # The remaining 'remaining_n' numbers will be a sequence starting from 'start_val'.
            first_rem = start_val
            last_rem = start_val + remaining_n - 1
            
            # Sum of this arithmetic progression: count * (first + last) / 2
            sum_rem = remaining_n * (first_rem + last_rem) // 2
            
            # Add the sum of remaining numbers to the total current sum.
            current_sum = (current_sum + sum_rem)

        # Apply modulo at the very end as per problem statement.
        return current_sum % MOD