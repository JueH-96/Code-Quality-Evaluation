class Solution:
    def countSymmetricIntegers(self, low: int, high: int) -> int:
        def is_symmetric(num):
            num_str = str(num)
            n = len(num_str)
            if n % 2 != 0:
                return False
            mid = n // 2
            return sum(int(digit) for digit in num_str[:mid]) == sum(int(digit) for digit in num_str[mid:])
        
        count = 0
        for num in range(low, high + 1):
            if is_symmetric(num):
                count += 1
        return count