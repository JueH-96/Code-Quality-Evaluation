class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # Helper function to check if a number is prime
        def is_prime(n):
            if n < 2:
                return False
            for i in range(2, int(n ** 0.5) + 1):
                if n % i == 0:
                    return False
            return True
        
        # Find all the prime numbers in the array
        primes = [num for num in nums if is_prime(num)]
        
        # If there is only one prime number, return 0
        if len(primes) == 1:
            return 0
        
        # Find the maximum distance between the indices of two prime numbers
        max_distance = 0
        for i in range(len(primes)):
            for j in range(i + 1, len(primes)):
                max_distance = max(max_distance, abs(nums.index(primes[i]) - nums.index(primes[j])))
        
        return max_distance