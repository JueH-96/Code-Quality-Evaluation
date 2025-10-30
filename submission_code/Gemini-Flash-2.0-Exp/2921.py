class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7

        def is_stepping(num_str):
            for i in range(len(num_str) - 1):
                if abs(int(num_str[i]) - int(num_str[i+1])) != 1:
                    return False
            return True

        def count_stepping_numbers_less_than_or_equal_to(limit):
            n = len(limit)
            dp = {}

            def solve(index, prev_digit, is_tight, is_leading_zero):
                if index == n:
                    return 1 if not is_leading_zero else 0

                if (index, prev_digit, is_tight, is_leading_zero) in dp:
                    return dp[(index, prev_digit, is_tight, is_leading_zero)]

                count = 0
                upper_bound = int(limit[index]) if is_tight else 9

                for digit in range(0, upper_bound + 1):
                    if is_leading_zero and digit == 0:
                        count = (count + solve(index + 1, -1, is_tight and (digit == upper_bound), True)) % MOD
                    elif not is_leading_zero:
                        if prev_digit == -1 or abs(digit - prev_digit) == 1:
                            count = (count + solve(index + 1, digit, is_tight and (digit == upper_bound), False)) % MOD
                    else:
                        count = (count + solve(index + 1, digit, is_tight and (digit == upper_bound), False)) % MOD

                dp[(index, prev_digit, is_tight, is_leading_zero)] = count
                return count

            result = 0
            for i in range(1, n):
                result = (result + 9 * pow(2, i - 1, MOD)) % MOD
            result = (result + solve(0, -1, True, True)) % MOD
            return result

        low_count = count_stepping_numbers_less_than_or_equal_to(low)
        high_count = count_stepping_numbers_less_than_or_equal_to(high)

        if is_stepping(low):
            return (high_count - low_count + 1 + MOD) % MOD
        else:
            return (high_count - low_count + MOD) % MOD