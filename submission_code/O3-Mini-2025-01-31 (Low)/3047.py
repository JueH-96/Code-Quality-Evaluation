from typing import List
import math

class Solution:
    def maximumSum(self, nums: List[int]) -> int:
        # Precompute primes up to sqrt(max(nums))
        # max(nums) can be 1e9 so sqrt is about 31623
        def sieve(n):
            is_prime = [True]*(n+1)
            is_prime[0] = is_prime[1] = False
            primes = []
            for i in range(2, n+1):
                if is_prime[i]:
                    primes.append(i)
                    for j in range(i*i, n+1, i):
                        is_prime[j] = False
            return primes

        primes = sieve(31623)
        
        # Function to compute the square-free part of a number
        def square_free(x):
            # We iterate over primes and for each we check exponent mod 2.
            # If exponent is odd then multiply factor.
            sq_free = 1
            temp = x
            for p in primes:
                if p*p > temp:
                    break
                count = 0
                while temp % p == 0:
                    count += 1
                    temp //= p
                if count % 2 == 1:
                    sq_free *= p
            # if after division temp > 1, then temp is prime or a product of primes (but in our logic it must be prime)
            if temp > 1:
                # It appears with exponent 1 which is odd
                sq_free *= temp
            return sq_free

        # Group numbers by their square-free representation.
        group_sum = {}
        group_count = {}
        max_single = 0
        for num in nums:
            max_single = max(max_single, num)
            key = square_free(num)
            group_sum[key] = group_sum.get(key, 0) + num
            group_count[key] = group_count.get(key, 0) + 1
        
        best = max_single
        # If a group has at least 2 numbers, then the whole group forms a complete subset.
        for key in group_sum:
            if group_count[key] >= 2:
                best = max(best, group_sum[key])
        return best

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.maximumSum([8,7,3,5,7,2,4,9]))  # Expected output: 16
    print(sol.maximumSum([5,10,3,10,1,13,7,9,4]))  # Expected output: 19