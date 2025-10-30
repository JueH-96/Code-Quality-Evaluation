import math

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        n = len(nums)
        first_digits = [0] * n
        last_digits = [0] * n
        
        for i, num in enumerate(nums):
            last_digits[i] = num % 10
            first_digits[i] = int(str(num)[0])
        
        res = 0
        for j in range(1, n):
            for i in range(j):
                if math.gcd(first_digits[i], last_digits[j]) == 1:
                    res += 1
        return res