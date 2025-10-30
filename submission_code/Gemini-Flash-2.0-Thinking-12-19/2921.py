class Solution:
    def countSteppingNumbers(self, low: str, high: str) -> int:
        MOD = 10**9 + 7

        def count_stepping_up_to(num_str):
            n = len(num_str)
            memo = {}

            def solve_tight(index, last_digit, tight):
                if index == n:
                    return 1
                if (index, last_digit, tight) in memo:
                    return memo[(index, last_digit, tight)]

                count = 0
                limit = int(num_str[index]) if tight else 9
                for digit in range(limit + 1):
                    if index == 0 and digit == 0:
                        continue
                    if index > 0 and abs(digit - last_digit) != 1:
                        continue
                    count += solve_tight(index + 1, digit, tight and (digit == limit))
                memo[(index, last_digit, tight)] = count
                return count

            ans = 0
            for length in range(1, n):
                ans += count_stepping_length(length)

            ans += solve_tight(0, -1, 1)
            return ans

        def count_stepping_length(length):
            count = 0
            memo = {}
            def generate_stepping_count(remaining_length, last_digit):
                if remaining_length == 0:
                    return 1
                if (remaining_length, last_digit) in memo:
                    return memo[(remaining_length, last_digit)]
                res = 0
                if last_digit > 0:
                    res += generate_stepping_count(remaining_length - 1, last_digit - 1)
                if last_digit < 9:
                    res += generate_stepping_count(remaining_length - 1, last_digit + 1)
                memo[(remaining_length, last_digit)] = res
                return res

            for start_digit in range(1, 10):
                count += generate_stepping_count(length - 1, start_digit)
            return count

        count_high = count_stepping_up_to(high)
        count_low = 0
        if int(low) > 1:
            count_low = count_stepping_up_to(str(int(low) - 1))
        
        return (count_high - count_low) % MOD