class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def is_beautiful(num: int, k: int) -> bool:
            # Convert the number to a string to count digits
            num_str = str(num)
            even_count = sum(1 for digit in num_str if int(digit) % 2 == 0)
            odd_count = len(num_str) - even_count
            return even_count == odd_count and num % k == 0

        count = 0
        for num in range(low, high + 1):
            if is_beautiful(num, k):
                count += 1
        return count