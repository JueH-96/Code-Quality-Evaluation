from math import gcd
from typing import List

def lcm(a: int, b: int) -> int:
    return a * b // gcd(a, b) if a and b else 0

class Solution:
    def maxLength(self, nums: List[int]) -> int:
        n = len(nums)
        ans = 0
        
        # helper to compute product, gcd, and lcm of a slice arr
        def check_subarray(arr: List[int]) -> bool:
            length = len(arr)
            # For length 1, product eq means a == a*a, so only possible if a==1.
            if length == 1:
                return arr[0] == 1
            # For length 2, it's always true (Property: a*b = gcd(a,b)*lcm(a,b) for any two positive integers)
            if length == 2:
                return True
            # For length >= 3, condition becomes:
            # prod(arr) == gcd(arr)*lcm(arr)
            # Note: In any subarray, let d=gcd(arr). Then writing a_i = d * b_i with gcd(b_i)=1,
            # we get product = d^n * (∏b_i) and lcm = d*lcm(b_i).
            # Hence, product == gcd* lcm  <=> d^(n-2)*(∏b_i) == lcm(b_i).
            # And because for pairwise coprime b_i we have lcm(b_i)=∏b_i, d must be 1.
            # Therefore, for n>=3, we need gcd(arr)==1 and also product == lcm(arr).
            current_gcd = arr[0]
            current_lcm = arr[0]
            prod = arr[0]
            for a in arr[1:]:
                current_gcd = gcd(current_gcd, a)
                current_lcm = lcm(current_lcm, a)
                prod *= a
            return current_gcd == 1 and prod == current_lcm
            
        # Brute-force check all subarrays
        for start in range(n):
            # For performance, we can compute in a running manner.
            current_gcd = 0
            current_lcm = 1
            prod = 1
            for end in range(start, n):
                a = nums[end]
                prod *= a
                current_gcd = a if current_gcd == 0 else gcd(current_gcd, a)
                current_lcm = lcm(current_lcm, a)
                length = end - start + 1
                # check condition for this subarray
                # Special handling for length 1:
                if length == 1:
                    valid = (a == 1)
                elif length == 2:
                    valid = True  # Always holds for two numbers
                else:
                    valid = (current_gcd == 1 and prod == current_lcm)
                if valid:
                    ans = max(ans, length)
        return ans