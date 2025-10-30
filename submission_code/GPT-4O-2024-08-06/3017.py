class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def is_beautiful(num: int) -> bool:
            even_count = 0
            odd_count = 0
            for digit in str(num):
                if int(digit) % 2 == 0:
                    even_count += 1
                else:
                    odd_count += 1
            return even_count == odd_count and num % k == 0

        beautiful_count = 0
        for num in range(low, high + 1):
            if is_beautiful(num):
                beautiful_count += 1

        return beautiful_count