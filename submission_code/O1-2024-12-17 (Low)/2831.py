class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        from math import gcd
        
        def first_digit(x: int) -> int:
            while x >= 10:
                x //= 10
            return x
        
        def last_digit(x: int) -> int:
            return x % 10
        
        count = 0
        n = len(nums)
        
        for i in range(n):
            f = first_digit(nums[i])
            for j in range(i+1, n):
                l = last_digit(nums[j])
                if gcd(f, l) == 1:
                    count += 1
        
        return count