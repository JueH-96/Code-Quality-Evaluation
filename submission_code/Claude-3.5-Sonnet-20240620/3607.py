class Solution:
    def minOperations(self, nums: List[int]) -> int:
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a

        def largest_proper_divisor(n):
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return n // i
            return 1

        n = len(nums)
        operations = 0
        
        # Handle the case where all numbers are 1
        if all(num == 1 for num in nums):
            return 0
        
        # Reduce all numbers to their GCD
        current_gcd = nums[0]
        for num in nums[1:]:
            current_gcd = gcd(current_gcd, num)
        
        if current_gcd > 1:
            return -1
        
        # Count operations needed to make the array non-decreasing
        for i in range(n - 1, 0, -1):
            while nums[i] < nums[i-1]:
                divisor = largest_proper_divisor(nums[i-1])
                if divisor == 1:
                    return -1
                nums[i-1] //= divisor
                operations += 1
        
        return operations