class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        primes = []
        for i, num in enumerate(nums):
            is_prime = True
            if num <= 1:
                is_prime = False
            else:
                for j in range(2, int(num**0.5) + 1):
                    if num % j == 0:
                        is_prime = False
                        break
            if is_prime:
                primes.append(i)
        
        if not primes:
            return 0
        
        if len(primes) == 1:
            return 0
        
        max_diff = 0
        for i in range(len(primes)):
            for j in range(len(primes)):
                max_diff = max(max_diff, abs(primes[i] - primes[j]))
        
        return max_diff