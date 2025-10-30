class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def is_beautiful(num, k):
            s_num = str(num)
            even_count = 0
            odd_count = 0
            for digit in s_num:
                digit = int(digit)
                if digit % 2 == 0:
                    even_count += 1
                else:
                    odd_count += 1
            return even_count == odd_count and num % k == 0

        count = 0
        for num in range(low, high + 1):
            if is_beautiful(num, k):
                count += 1
        return count