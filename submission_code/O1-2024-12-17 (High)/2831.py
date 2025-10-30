class Solution:
    def countBeautifulPairs(self, nums: List[int]) -> int:
        from math import gcd
        
        def first_digit(x: int) -> int:
            # Extract the first digit of x
            while x >= 10:
                x //= 10
            return x
        
        count = 0
        n = len(nums)
        
        for i in range(n):
            f_i = first_digit(nums[i])
            for j in range(i+1, n):
                # Last digit of nums[j]
                l_j = nums[j] % 10
                # Check if they are coprime
                if gcd(f_i, l_j) == 1:
                    count += 1
        
        return count