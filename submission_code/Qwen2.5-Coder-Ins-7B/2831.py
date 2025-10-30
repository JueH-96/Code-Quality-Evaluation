class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        count = 0
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if self.gcd(self.first_digit(nums[i]), nums[j] % 10) == 1:
                    count += 1
        return count
    
    def first_digit(self, n):
        while n >= 10:
            n //= 10
        return n
    
    def gcd(self, a, b):
        while b:
            a, b = b, a % b
        return a