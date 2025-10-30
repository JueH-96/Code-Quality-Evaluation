class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def count_even_odd(n):
            even_count = 0
            odd_count = 0
            for digit in str(n):
                digit = int(digit)
                if digit % 2 == 0:
                    even_count += 1
                else:
                    odd_count += 1
            return even_count, odd_count

        count = 0
        for i in range(low, high + 1):
            even, odd = count_even_odd(i)
            if even == odd and i % k == 0:
                count += 1
        return count