class Solution:
    def minOperations(self, nums: List[int]) -> int:
        def smallest_prime_factor(n):
            if n <= 1:
                return n
            if n % 2 == 0:
                return 2
            for i in range(3, int(n**0.5) + 1, 2):
                if n % i == 0:
                    return i
            return n
        
        def greatest_proper_divisor(n):
            spf = smallest_prime_factor(n)
            if spf == n:  # n is prime
                return 1
            return n // spf
        
        n = len(nums)
        operations = 0
        
        # Work from right to left
        for i in range(n - 2, -1, -1):
            # While current element is greater than next element
            while nums[i] > nums[i + 1]:
                gpd = greatest_proper_divisor(nums[i])
                if gpd == 1:  # nums[i] is prime, can't reduce further
                    return -1
                nums[i] = nums[i] // gpd
                operations += 1
        
        return operations