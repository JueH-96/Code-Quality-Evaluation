class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        # Define num1 as the sum of integers in [1, n] not divisible by m.
        # Define num2 as the sum of integers in [1, n] divisible by m.
        # We need to return num1 - num2.

        # Approach 1: Iterate through all numbers from 1 to n.
        # Initialize sums
        num1 = 0
        num2 = 0

        # Iterate from 1 up to and including n
        for i in range(1, n + 1):
            # Check if the current number i is divisible by m
            if i % m == 0:
                # If divisible, add it to num2
                num2 += i
            else:
                # If not divisible, add it to num1
                num1 += i

        # Return the difference
        return num1 - num2

        # Approach 2 (Optimized): Use mathematical formulas.
        # The sum of all integers from 1 to n is sum_total = n * (n + 1) // 2.
        # We know that sum_total = num1 + num2.
        # We want to find num1 - num2.
        # Substituting num1 = sum_total - num2 into the expression:
        # num1 - num2 = (sum_total - num2) - num2 = sum_total - 2 * num2.
        # So, we only need to efficiently calculate sum_total and num2.

        # Calculate sum_total
        # sum_total = n * (n + 1) // 2 # Using integer division

        # Calculate num2 (sum of integers in [1, n] divisible by m).
        # The integers divisible by m in the range [1, n] are m, 2m, 3m, ..., km,
        # where k is the largest integer such that km <= n.
        # k is found by integer division: k = n // m.
        # num2 = m*1 + m*2 + ... + m*k
        # num2 = m * (1 + 2 + ... + k)
        # The sum 1 + 2 + ... + k is k * (k + 1) // 2.
        # Let k = n // m.
        # sum_of_multiples_up_to_k = k * (k + 1) // 2
        # num2 = m * sum_of_multiples_up_to_k

        # So, the optimized calculation for num2 is:
        # k = n // m
        # num2_optimized = m * (k * (k + 1) // 2)

        # Calculate sum_total - 2 * num2_optimized
        # result_optimized = sum_total - 2 * num2_optimized

        # Return result_optimized
        # This optimized approach is not necessary given the constraints (n <= 1000),
        # the simple iterative approach is clear and efficient enough.
        # Keeping the iterative approach as it directly follows the definition.
        # return (n * (n + 1) // 2) - 2 * (m * ((n // m) * (n // m + 1) // 2))