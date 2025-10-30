class Solution:
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        def is_beautiful(n):
            even = 0
            odd = 0
            for digit in str(n):
                if int(digit) % 2 == 0:
                    even += 1
                else:
                    odd += 1
            return even == odd and n % k == 0
        
        count = 0
        for num in range(low, high + 1):
            if is_beautiful(num):
                count += 1
        return count