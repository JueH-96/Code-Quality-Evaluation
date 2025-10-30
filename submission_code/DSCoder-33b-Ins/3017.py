class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def is_beautiful(num):
            even_count = sum(1 for d in str(num) if int(d) % 2 == 0)
            odd_count = sum(1 for d in str(num) if int(d) % 2 != 0)
            return even_count == odd_count and num % k == 0

        return sum(1 for num in range(low, high + 1) if is_beautiful(num))