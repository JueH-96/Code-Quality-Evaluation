class Solution:
    def isBeautifulNumber(self, num: int, k: int) -> bool:
        even_digits_count = 0
        odd_digits_count = 0
        temp_num = num
        if temp_num == 0:
            even_digits_count = 1
        else:
            while temp_num > 0:
                digit = temp_num % 10
                if digit % 2 == 0:
                    even_digits_count += 1
                else:
                    odd_digits_count += 1
                temp_num //= 10
        if even_digits_count == odd_digits_count and num % k == 0:
            return True
        else:
            return False
            
    def numberOfBeautifulIntegers(self, low: int, high: int, k: int) -> int:
        count = 0
        for num in range(low, high + 1):
            if self.isBeautifulNumber(num, k):
                count += 1
        return count