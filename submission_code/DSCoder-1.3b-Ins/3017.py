class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count_digits(n):
            return len([d for d in str(n) if d != '0' and n % d == 0])

        def is_beautiful(n):
            return count_digits(n) % 2 == 0 and count_digits(n) != 0 and n % k == 0

        count = 0
        for i in range(low, high + 1):
            if is_beautiful(i):
                count += 1

        return count