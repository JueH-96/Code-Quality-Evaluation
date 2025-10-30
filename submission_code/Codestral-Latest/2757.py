class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        def digit_sum(n):
            return sum(int(d) for d in str(n))

        def count_good_integers(low, high, min_sum, max_sum):
            count = 0
            for num in range(low, high + 1):
                if min_sum <= digit_sum(num) <= max_sum:
                    count += 1
            return count

        # Convert num1 and num2 to integers
        num1_int = int(num1)
        num2_int = int(num2)

        # Count good integers in the range [num1, num2]
        result = count_good_integers(num1_int, num2_int, min_sum, max_sum)

        return result % MOD