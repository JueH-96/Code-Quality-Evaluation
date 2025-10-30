import math

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        first_digits = []
        last_digits = []
        for num in nums:
            first = int(str(num)[0])
            last = num % 10
            first_digits.append(first)
            last_digits.append(last)
        count = 0
        for i in range(len(first_digits)):
            for j in range(i + 1, len(last_digits)):
                a = first_digits[i]
                b = last_digits[j]
                if math.gcd(a, b) == 1:
                    count += 1
        return count