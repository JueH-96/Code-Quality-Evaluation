class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        
        # num1: The sum of all integers in the range [1, n] that are not divisible by m.
        # num2: The sum of all integers in the range [1, n] that are divisible by m.
        # We need to return num1 - num2.

        # Calculate S_total = sum of all integers from 1 to n.
        # The formula for the sum of the first n integers is n * (n + 1) / 2.
        # Using integer division for safety, although result is always integer if n*(n+1) is even.
        sum_total = n * (n + 1) // 2
        
        # Calculate num2: sum of integers in [1, n] that are divisible by m.
        # These integers are m, 2*m, 3*m, ..., p*m, where p*m is the largest multiple of m <= n.
        # The largest integer p such that p*m <= n is p = n // m.
        p = n // m
        
        # The sum of the series 1, 2, ..., p is p * (p + 1) / 2.
        sum_of_first_p_integers = p * (p + 1) // 2
        
        # So, num2 = m * (1 + 2 + ... + p) = m * sum_of_first_p_integers.
        num2 = m * sum_of_first_p_integers
        
        # num1 is the sum of integers in [1, n] that are not divisible by m.
        # Since any integer in [1, n] is either divisible by m or not,
        # it must be that num1 + num2 = sum_total.
        # Therefore, num1 = sum_total - num2.
        num1 = sum_total - num2
        
        # The function is required to return the integer num1 - num2.
        return num1 - num2