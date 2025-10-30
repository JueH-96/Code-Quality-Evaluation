class Solution:
    def minOperations(self, nums: List[int]) -> int:
        def smallest_prime_factor(n):
            if n <= 1:
                return n
            for i in range(2, int(n**0.5) + 1):
                if n % i == 0:
                    return i
            return n  # n is prime
        
        operations = 0
        n = len(nums)
        
        # Process from right to left
        for i in range(n - 2, -1, -1):
            while nums[i] > nums[i + 1]:
                spf = smallest_prime_factor(nums[i])
                if spf == nums[i]:  # nums[i] is prime and > nums[i+1]
                    return -1
                nums[i] = spf  # Replace with smallest prime factor
                operations += 1
        
        return operations