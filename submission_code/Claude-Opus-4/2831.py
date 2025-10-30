class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        def get_first_digit(n):
            while n >= 10:
                n //= 10
            return n
        
        def get_last_digit(n):
            return n % 10
        
        count = 0
        n = len(nums)
        
        for i in range(n):
            for j in range(i + 1, n):
                first_digit = get_first_digit(nums[i])
                last_digit = get_last_digit(nums[j])
                
                if gcd(first_digit, last_digit) == 1:
                    count += 1
        
        return count