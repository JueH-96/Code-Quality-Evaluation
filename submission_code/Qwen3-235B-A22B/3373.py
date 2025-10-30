class Solution:
    def maximumPrimeDifference(self, nums: List[int]) -> int:
        # Precompute the set of primes up to 100 using sieve of Eratosthenes
        is_prime = [True] * 101
        is_prime[0] = is_prime[1] = False
        for i in range(2, 101):
            if is_prime[i]:
                # Mark multiples of i as non-prime
                for j in range(i * i, 101, i):
                    is_prime[j] = False
        
        # Find first and last occurrence of primes in nums
        first = -1
        last = -1
        for idx, num in enumerate(nums):
            if is_prime[num]:
                if first == -1:
                    first = idx
                last = idx
        
        return last - first