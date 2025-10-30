class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def is_beautiful(num: int) -> bool:
            num_str = str(num)
            even_count = sum(1 for digit in num_str if int(digit) % 2 == 0)
            odd_count = sum(1 for digit in num_str if int(digit) % 2 != 0)
            return even_count == odd_count and num % k == 0
        
        return sum(is_beautiful(num) for num in range(low, high + 1))