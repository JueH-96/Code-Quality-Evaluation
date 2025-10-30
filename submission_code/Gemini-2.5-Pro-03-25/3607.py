import math
from typing import List

# --- Sieve Implementation Start ---
# Precompute Smallest Prime Factor (SPF) for numbers up to MAX_VAL
# The maximum value in nums is 10^6, so we need an array of size 10^6 + 1.
MAX_VAL = 10**6 + 1 
# Initialize spf array where spf[i] will store the smallest prime factor of i.
spf = [i for i in range(MAX_VAL)]

# Define the sieve function to precompute SPF values.
# This function should be called once before any calls to minOperations.
def _sieve(): 
    """ Precomputes SPF for all numbers up to MAX_VAL using a sieve. """
    # 1 is not prime, its SPF is conventionally set to 1.
    spf[1] = 1
    # Iterate from 2 up to sqrt(MAX_VAL).
    limit = int(math.sqrt(MAX_VAL)) + 1
    for i in range(2, limit):
        # If i is prime (its SPF is still i)
        if spf[i] == i: 
            # Mark the SPF for all multiples of i.
            # Start marking from i*i, as smaller multiples would have already been marked by smaller primes.
            for j in range(i*i, MAX_VAL, i):
                # Update spf[j] only if it hasn't been set by a smaller prime factor yet.
                # This ensures spf[j] always stores the *smallest* prime factor.
                if spf[j] == j: 
                    spf[j] = i

# Execute the sieve function once when the module is loaded to precompute SPF values.
_sieve()
# --- Sieve Implementation End ---


class Solution:
    
    # Helper method to determine the value of x after one transformation operation.
    # It uses the precomputed global `spf` array.
    def get_transformed_value(self, x: int) -> int:
        """ 
        Calculates the value after one operation based on the problem definition.
        If x is composite, its greatest proper divisor is x / SPF(x). 
        The operation replaces x with x / (x / SPF(x)) = SPF(x).
        If x is prime, its greatest proper divisor is 1. The operation replaces x with x / 1 = x.
        If x = 1, it has no proper divisors, operation is not possible/doesn't change value.
        """
        if x <= 1: 
            # Base case: 1 cannot be transformed.
            return 1
        # Check if x is composite using the precomputed SPF table. 
        # If spf[x] != x, then x is composite.
        if spf[x] != x: 
            # If composite, the transformed value is its smallest prime factor.
            return spf[x] 
        else: 
            # If prime (spf[x] == x), the value remains unchanged.
            return x

    # Helper method to determine the cost of one transformation operation on x.
    # Cost is 1 if the operation changes the value (i.e., if x is composite), otherwise 0.
    def get_cost_of_transforming(self, x: int) -> int:
        """
        Calculates the cost of one operation on x.
        Returns 1 if x is composite.
        Returns 0 if x is prime or 1.
        """
        if x <= 1: 
            # No operation cost for 1.
            return 0
        # The operation costs 1 only if x is composite.
        if spf[x] != x: 
            # Cost 1 for composite numbers.
            return 1 
        else: 
            # Cost 0 for prime numbers (as operation doesn't change them).
            return 0 

    def minOperations(self, nums: List[int]) -> int:
        """
        Calculates the minimum number of operations to make the array nums non-decreasing.
        Uses dynamic programming with state compression (O(1) space for DP table).
        """
        n = len(nums)
        # An empty array is considered non-decreasing with 0 operations.
        if n == 0:
            return 0
        
        # Use float('inf') to represent unreachable states in DP.
        infinity = float('inf')

        # Initialize DP states based on the first element nums[0].
        # We need to track the minimum operations ending with nums[i] either unchanged or transformed.
        
        # v0_prev: Value of the previous element (nums[i-1]) if it was kept unchanged.
        v0_prev = nums[0]  
        # v1_prev: Value of the previous element (nums[i-1]) if it was transformed.
        v1_prev = self.get_transformed_value(nums[0]) 
        # cost0: Cost incurred to transform nums[0] (1 if composite, 0 otherwise).
        cost0 = self.get_cost_of_transforming(nums[0]) 
        
        # dp0_prev: Minimum operations for the prefix ending at index i-1, where nums[i-1] has value v0_prev.
        dp0_prev = 0      
        # dp1_prev: Minimum operations for the prefix ending at index i-1, where nums[i-1] has value v1_prev.
        # This state costs `cost0` operations initially at index 0.
        dp1_prev = cost0  

        # Iterate through the array starting from the second element (index 1).
        for i in range(1, n):
            curr_val = nums[i]
            # curr_v1: Value of nums[i] if kept unchanged.
            curr_v1 = curr_val  
            # curr_v2: Value of nums[i] if transformed.
            curr_v2 = self.get_transformed_value(curr_val) 
            # cost_curr: Cost to transform nums[i] (1 if composite, 0 otherwise).
            cost_curr = self.get_cost_of_transforming(curr_val) 

            # Initialize minimum operations for the current index i to infinity.
            # dp0_curr: Min ops for prefix ending at index i, with nums[i] kept as original value (curr_v1).
            dp0_curr = infinity 
            # dp1_curr: Min ops for prefix ending at index i, with nums[i] transformed (curr_v2).
            dp1_curr = infinity 

            # Calculate dp0_curr: minimum operations to end with current element unchanged (curr_v1).
            # Check transition from previous state ending with v0_prev (unchanged).
            # The non-decreasing condition requires curr_v1 >= v0_prev.
            if curr_v1 >= v0_prev and dp0_prev != infinity:
                dp0_curr = min(dp0_curr, dp0_prev) 
            # Check transition from previous state ending with v1_prev (transformed).
            # The non-decreasing condition requires curr_v1 >= v1_prev.
            if curr_v1 >= v1_prev and dp1_prev != infinity:
                dp0_curr = min(dp0_curr, dp1_prev)

            # Calculate dp1_curr: minimum operations to end with current element transformed (curr_v2).
            # This transition incurs an additional cost `cost_curr` if the transformation happens.
            # Check transition from previous state ending with v0_prev.
            # The non-decreasing condition requires curr_v2 >= v0_prev.
            if curr_v2 >= v0_prev and dp0_prev != infinity:
                dp1_curr = min(dp1_curr, dp0_prev + cost_curr)
            # Check transition from previous state ending with v1_prev.
            # The non-decreasing condition requires curr_v2 >= v1_prev.
            if curr_v2 >= v1_prev and dp1_prev != infinity:
                dp1_curr = min(dp1_curr, dp1_prev + cost_curr)

            # Update the DP states for the next iteration (i+1).
            # The current states become the previous states for the next element.
            dp0_prev = dp0_curr
            dp1_prev = dp1_curr
            # Update the potential values associated with the DP states for the next element check.
            v0_prev = curr_v1 
            v1_prev = curr_v2

        # After iterating through the array, the final minimum operations is the minimum of 
        # the costs to reach either final state for the last element nums[n-1].
        final_min_ops = min(dp0_prev, dp1_prev)

        # If the minimum cost is still infinity, it means it's impossible to make the array non-decreasing.
        if final_min_ops == infinity:
            return -1 
        else:
            # Otherwise, return the calculated minimum number of operations.
            return final_min_ops