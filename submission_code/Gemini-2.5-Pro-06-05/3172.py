class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        """
        Calculates the difference between the sum of numbers not divisible by m
        and the sum of numbers divisible by m in the range [1, n].
        
        The problem asks for `num1 - num2`. This can be simplified to:
        `total_sum - 2 * num2`, where `total_sum` is the sum of all integers
        from 1 to n, and `num2` is the sum of integers divisible by m.
        """
        
        # Calculate the sum of all integers from 1 to n using the formula for
        # the sum of an arithmetic series.
        total_sum = n * (n + 1) // 2
        
        # The integers divisible by m are m, 2*m, ..., k*m, where k is the
        # number of multiples of m in the range [1, n].
        k = n // m
        
        # Calculate the sum of these multiples, which is num2.
        # num2 = m * (1 + 2 + ... + k)
        sum_of_multiples = m * (k * (k + 1) // 2)
        
        # The final result is total_sum - 2 * num2.
        return total_sum - 2 * sum_of_multiples