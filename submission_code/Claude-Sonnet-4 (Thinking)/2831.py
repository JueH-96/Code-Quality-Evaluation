class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        from math import gcd
        
        def get_first_digit(n):
            while n >= 10:
                n //= 10
            return n
        
        count = 0
        n = len(nums)
        
        for i in range(n):
            for j in range(i + 1, n):
                first_digit = get_first_digit(nums[i])
                last_digit = nums[j] % 10
                
                if gcd(first_digit, last_digit) == 1:
                    count += 1
        
        return count