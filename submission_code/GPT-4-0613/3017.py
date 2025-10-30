class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def is_beautiful(n: int) -> bool:
            str_n = str(n)
            even_count = sum(1 for digit in str_n if int(digit) % 2 == 0)
            odd_count = len(str_n) - even_count
            return even_count == odd_count and n % k == 0

        return sum(1 for n in range(low, high + 1) if is_beautiful(n))