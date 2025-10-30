class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        def first_digit(num):
            while num >= 10:
                num //= 10
            return num

        count = 0
        n = len(nums)
        
        for i in range(n):
            for j in range(i + 1, n):
                if gcd(first_digit(nums[i]), nums[j] % 10) == 1:
                    count += 1
        
        return count