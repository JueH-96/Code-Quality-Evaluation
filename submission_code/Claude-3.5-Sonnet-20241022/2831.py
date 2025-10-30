class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
            
        def get_first_digit(num):
            while num >= 10:
                num //= 10
            return num
            
        count = 0
        n = len(nums)
        
        for i in range(n):
            first = get_first_digit(nums[i])
            for j in range(i+1, n):
                last = nums[j] % 10
                if gcd(first, last) == 1:
                    count += 1
                    
        return count