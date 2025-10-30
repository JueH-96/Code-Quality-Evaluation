import math
from typing import List

class Solution:
    def _is_product_equivalent(self, arr: List[int]) -> bool:
        k = len(arr)

        # Base case: empty array is not a valid subarray (problem implies non-empty)
        if k == 0:
            return False
        
        # Case 1: Subarray of length 1
        # prod([a]) = a
        # gcd([a]) = a
        # lcm([a]) = a
        # Condition: a == a * a  => a^2 - a = 0 => a(a-1) = 0
        # Since nums[i] are positive, a must be 1.
        if k == 1:
            return arr[0] == 1
        
        # Case 2: Subarray of length 2
        # prod([a, b]) = a * b
        # gcd([a, b]) = math.gcd(a, b)
        # lcm([a, b]) = (a * b) // math.gcd(a, b)
        # Condition: a * b == ((a * b) // math.gcd(a, b)) * math.gcd(a, b)
        # This simplifies to a * b == a * b, which is always true.
        if k == 2:
            return True
        
        # Case 3: Subarray of length k > 2
        # Based on mathematical derivation, prod(arr) == lcm(arr) * gcd(arr) holds
        # if and only if for every prime p, sum(v_p(ai)) = min(v_p(ai)) + max(v_p(ai)).
        # For k > 2, this further implies that for every prime p,
        # at most one element in the array is divisible by p.
        # (Meaning, for a prime p, v_p(x) > 0 for at most one x in arr)
        
        # Primes relevant for numbers up to 10 are 2, 3, 5, 7.
        primes = [2, 3, 5, 7]
        
        for p in primes:
            count_divisible_by_p = 0
            for x in arr:
                # If x is divisible by p, then v_p(x) > 0
                if x % p == 0:
                    count_divisible_by_p += 1
            
            # If more than one element is divisible by the current prime p,
            # the condition is not met.
            if count_divisible_by_p > 1:
                return False
                
        return True

    def maxLength(self, nums: List[int]) -> int:
        n = len(nums)
        max_len = 0

        # Iterate over all possible subarrays
        # i is the starting index, j is the ending index
        for i in range(n):
            for j in range(i, n):
                sub_arr = nums[i:j+1]
                if self._is_product_equivalent(sub_arr):
                    max_len = max(max_len, len(sub_arr))
        
        # As per constraints, nums.length >= 2.
        # Any subarray of length 2 is product equivalent, so max_len will be at least 2.
        # If nums = [2,2], the loop will find [2,2] which is length 2 and product equivalent.
        # Initializing max_len = 0 is safe as it will be updated.
        return max_len