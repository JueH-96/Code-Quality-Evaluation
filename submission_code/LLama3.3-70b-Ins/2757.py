class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        def digit_sum(x: int) -> int:
            """Calculate the sum of digits of a number."""
            return sum(int(digit) for digit in str(x))

        def count_good_integers(num: str) -> int:
            """Count the number of good integers less than or equal to num."""
            count = 0
            for x in range(1, int(num) + 1):
                if min_sum <= digit_sum(x) <= max_sum:
                    count += 1
            return count

        # Calculate the number of good integers less than or equal to num2
        count_num2 = count_good_integers(num2)

        # Calculate the number of good integers less than num1
        count_num1 = count_good_integers(str(int(num1) - 1))

        # The number of good integers between num1 and num2 is the difference
        # between the counts of good integers less than or equal to num2 and num1
        return (count_num2 - count_num1) % MOD