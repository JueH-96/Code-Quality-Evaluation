class Solution:
    def count(self, num1: str, num2: str, min_sum: int, max_sum: int) -> int:
        MOD = 10**9 + 7

        # Helper: subtract 1 from a non-negative decimal string
        def sub_one(num_str: str) -> str:
            # Special case
            if num_str == "0":
                return "0"
            digits = list(map(int, num_str))
            i = len(digits) - 1
            while i >= 0:
                if digits[i] > 0:
                    digits[i] -= 1
                    break
                else:
                    digits[i] = 9
                    i -= 1
            # Remove leading zeros
            while len(digits) > 1 and digits[0] == 0:
                digits.pop(0)
            return "".join(map(str, digits))

        # Digit DP: counts how many numbers from 0..num_str (inclusive)
        # have digit sums in [min_sum, max_sum]
        def count_up_to(num_str: str, min_s: int, max_s: int) -> int:
            # Handle the corner case "0" easily:
            # if num_str == "0", then digit sum = 0, check if in range
            if num_str == "0":
                return 1 if (min_s <= 0 <= max_s) else 0

            digits = list(map(int, num_str))
            n = len(digits)
            # dp memo: key -> (pos, sum_so_far, is_tight)
            # value -> count of valid numbers
            from functools import lru_cache

            @lru_cache(None)
            def dfs(pos, sum_so_far, is_tight):
                if pos == n:
                    return 1 if (min_s <= sum_so_far <= max_s) else 0

                limit = digits[pos] if is_tight else 9
                res = 0
                for d in range(limit + 1):
                    new_sum = sum_so_far + d
                    if new_sum > max_s:
                        break  # No need to continue (sum would only get larger)
                    res += dfs(pos + 1, new_sum, is_tight and (d == limit))
                return res % MOD

            return dfs(0, 0, True)

        # Main logic: count valid numbers in [num1..num2] =
        # count_up_to(num2) - count_up_to(num1 - 1)
        num1_minus_one = sub_one(num1)
        ans = count_up_to(num2, min_sum, max_sum) - count_up_to(num1_minus_one, min_sum, max_sum)
        return ans % MOD