from typing import List

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        primes = [2, 3, 5, 7]
        
        # Precompute v_p(n) for n in [1, 10] and p in primes.
        # v_p(n) is the exponent of prime p in the factorization of n.
        v_table = {}
        for num in range(1, 11):
            exponents_for_num = {}
            for p in primes:
                count = 0
                temp_n = num
                while temp_n > 0 and temp_n % p == 0:
                    count += 1
                    temp_n //= p
                exponents_for_num[p] = count
            v_table[num] = exponents_for_num

        n = len(nums)
        max_len = 0
        
        # Iterate through all possible starting positions of a subarray
        for i in range(n):
            # For each starting position i, check all subarrays starting at i.
            
            # These track the sum, min, and max of exponents for each prime
            # for the subarray nums[i...j].
            sums = {p: 0 for p in primes}
            mins = {p: float('inf') for p in primes}
            maxs = {p: -float('inf') for p in primes}
            
            # Extend the subarray from i to j
            for j in range(i, n):
                new_num = nums[j]
                
                # 1. Update the state (sums, mins, maxs) for the new number.
                for p in primes:
                    v = v_table[new_num][p]
                    sums[p] += v
                    mins[p] = min(mins[p], v)
                    maxs[p] = max(maxs[p], v)
                
                # 2. Check if the new subarray nums[i..j] is product equivalent.
                # This is true if for every prime p, sum(exponents) == min(exponents) + max(exponents).
                is_valid = True
                for p in primes:
                    if sums[p] != mins[p] + maxs[p]:
                        is_valid = False
                        break
                
                if is_valid:
                    max_len = max(max_len, j - i + 1)
                    
        return max_len