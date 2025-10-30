class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count_beautiful_numbers(n, k):
            count = 0
            for num in range(n, 0, -1):
                if num % k == 0:
                    even_count, odd_count = 0, 0
                    for digit in str(num):
                        if int(digit) % 2 == 0:
                            even_count += 1
                        else:
                            odd_count += 1
                    if even_count == odd_count:
                        count += 1
            return count
        
        return count_beautiful_numbers(high, k) - count_beautiful_numbers(low - 1, k)