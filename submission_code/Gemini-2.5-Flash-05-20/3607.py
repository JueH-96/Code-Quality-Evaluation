import math
from typing import List

class Solution:
    # Class-level static variable for SPF array to precompute once.
    _spf = []
    _MAX_NUM = 10**6 + 1 # Maximum possible value for nums[i] is 10^6

    # Sieve to precompute Smallest Prime Factor (SPF) for all numbers up to _MAX_NUM.
    # This method runs only once when the class is first loaded or an instance is created.
    def __init__(self):
        if not Solution._spf:
            Solution._spf = list(range(Solution._MAX_NUM))
            # spf[1] remains 1, indicating it's not prime in this context, and can't be reduced.
            # spf[0] is not used since nums[i] >= 1.

            # Sieve of Eratosthenes to find SPF for each number
            # Optimization: iterate up to sqrt(MAX_NUM) for outer loop primes
            for i in range(2, int(math.sqrt(Solution._MAX_NUM)) + 1):
                # If spf[i] is still i, it means i is a prime number.
                if Solution._spf[i] == i:
                    # Mark multiples of i.
                    # Start from i*i because smaller multiples would have already been marked
                    # by smaller prime factors.
                    for j in range(i * i, Solution._MAX_NUM, i):
                        # Only update spf[j] if it hasn't been set by a smaller prime factor yet.
                        if Solution._spf[j] == j:
                            Solution._spf[j] = i
    
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        total_operations = 0
        
        # 'next_val' stores the target value for the current element (nums[i])
        # It's initialized with the last element's value as there's no element
        # to its right to constrain it.
        next_val = nums[n-1] 
        
        # Iterate from the second-to-last element down to the first element.
        for i in range(n - 2, -1, -1):
            curr_val = nums[i]
            
            # Case 1: Current element is already less than or equal to the next element's target value.
            if curr_val <= next_val:
                # No operations needed for curr_val.
                # To maximize flexibility for nums[i-1], we set next_val to curr_val.
                # This ensures that nums[i-1] needs to be <= curr_val (its original value),
                # not the potentially larger original next_val.
                next_val = curr_val
            # Case 2: Current element is greater than the next element's target value.
            else:
                # Operations are needed. Determine if curr_val can be reduced.
                
                # Check if curr_val is prime (or 1).
                # In our spf array, spf[prime_number] == prime_number.
                # spf[1] == 1 also correctly falls into this category (cannot be reduced).
                if Solution._spf[curr_val] == curr_val: 
                    # If curr_val is prime (or 1), it cannot be reduced by the operation.
                    # Since curr_val > next_val, and it cannot be changed, it's impossible
                    # to satisfy the non-decreasing condition.
                    return -1
                else: # curr_val is composite
                    # A composite number 'x' can be reduced to its smallest prime factor (spf[x])
                    # in exactly one operation.
                    reduced_val = Solution._spf[curr_val]
                    
                    # After one operation, check if the reduced value is still too large.
                    # 'reduced_val' will be a prime number (or 1), so it cannot be reduced further.
                    if reduced_val > next_val:
                        # Even after the only possible operation, it's still not small enough.
                        return -1
                    else:
                        # reduced_val <= next_val, so this transformation is valid.
                        total_operations += 1
                        # Update next_val for the next iteration (i.e., for nums[i-1])
                        # to the value that nums[i] will take.
                        next_val = reduced_val
                                
        # If the loop completes, it means a non-decreasing array could be formed.
        return total_operations