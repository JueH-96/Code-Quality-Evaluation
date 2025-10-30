class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count_beautiful_numbers(n, k):
            if n == 0:
                return 0
            str_n = str(n)
            length = len(str_n)
            memo = {}

            def dfs(index, is_limit, is_number, odd_count, even_count, remainder):
                if index == length:
                    return int(is_number and odd_count == even_count and remainder == 0)
                if not is_limit and is_number and (index, odd_count, even_count, remainder) in memo:
                    return memo[(index, odd_count, even_count, remainder)]
                total = 0
                if not is_number:
                    total += dfs(index + 1, False, False, odd_count, even_count, remainder)
                low = 1 if not is_number else 0
                up = int(str_n[index]) if is_limit else 9
                for digit in range(low, up + 1):
                    new_is_limit = is_limit and digit == up
                    new_is_number = is_number or digit > 0
                    new_odd_count = odd_count + (digit % 2 == 1)
                    new_even_count = even_count + (digit % 2 == 0)
                    new_remainder = (remainder * 10 + digit) % k
                    total += dfs(index + 1, new_is_limit, new_is_number, new_odd_count, new_even_count, new_remainder)
                if not is_limit and is_number:
                    memo[(index, odd_count, even_count, remainder)] = total
                return total

            return dfs(0, True, False, 0, 0, 0)

        return count_beautiful_numbers(high, k) - count_beautiful_numbers(low - 1, k)