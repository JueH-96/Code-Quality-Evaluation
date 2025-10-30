from math import gcd

class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        # Extract the first digit and last digit of each number
        first_digits = [int(str(num)[0]) for num in nums]
        last_digits = [num % 10 for num in nums]
        
        count = 0
        n = len(nums)
        
        for i in range(n):
            for j in range(i+1, n):
                if gcd(first_digits[i], last_digits[j]) == 1:
                    count += 1
        
        return count