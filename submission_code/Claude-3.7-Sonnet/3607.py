class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        operations = 0
        
        for i in range(n - 1):
            while nums[i] > nums[i + 1]:
                is_prime_or_one, gpd = self.check_and_get_gpd(nums[i])
                
                if is_prime_or_one:
                    return -1  # nums[i] is 1 or prime and greater than nums[i+1], so can't be reduced further
                
                nums[i] //= gpd
                operations += 1
        
        return operations
    
    def check_and_get_gpd(self, n):
        """
        Check if a number is 1 or a prime, and get its greatest proper divisor.
        Returns a tuple (is_prime_or_one, gpd).
        """
        if n == 1:
            return True, None  # 1 is a special case with no proper divisor
        
        for i in range(2, int(n**0.5) + 1):
            if n % i == 0:
                return False, n // i  # n is composite, return its greatest proper divisor
        
        return True, 1  # n is prime, its greatest proper divisor is 1