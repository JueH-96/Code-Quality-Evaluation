class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        from math import gcd
        
        # Helper functions to extract the first and last digits.
        def first_digit(x: int) -> int:
            while x >= 10:
                x //= 10
            return x
        
        def last_digit(x: int) -> int:
            return x % 10
        
        count = 0
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                if gcd(first_digit(nums[i]), last_digit(nums[j])) == 1:
                    count += 1
        return count