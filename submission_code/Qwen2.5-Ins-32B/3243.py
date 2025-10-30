class Solution:
    def numberOfPowerfulInt(self, start: int, finish: int, limit: int, s: str) -> int:
        def count Powerful(n: str, limit: int, suffix: str) -> int:
            if len(n) < len(suffix):
                return 0
            if len(n) == len(suffix):
                return int(n >= suffix and all(int(d) <= limit for d in n))
            total = 0
            for first_digit in range(1, limit + 1):
                total += count_Powerful(str(first_digit) + '0' * (len(n) - 1), limit, suffix)
            if int(n[0]) <= limit:
                total += count_Powerful(n[1:], limit, suffix)
            return total

        return count_Powerful(str(finish), limit, s) - count_Powerful(str(start - 1), limit, s)