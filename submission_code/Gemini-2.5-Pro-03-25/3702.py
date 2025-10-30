import math # Required for math.gcd
from typing import List # Required for type hinting

class Solution:
    # Helper function to compute the greatest common divisor (GCD) of two integers.
    # We use the built-in math.gcd for efficiency and correctness.
    def gcd(self, a: int, b: int) -> int:
        """Computes the greatest common divisor of a and b."""
        # math.gcd handles edge cases like gcd(a, 0) = abs(a).
        # Since the problem constraints state nums[i] >= 1, inputs a and b 
        # during the incremental calculation will always be positive integers >= 1.
        return math.gcd(a, b)

    # Helper function to compute the least common multiple (LCM) of two integers.
    def lcm(self, a: int, b: int) -> int:
        """Computes the least common multiple of a and b."""
        # Handle the case where a or b might be 0. 
        # Although based on nums[i] >= 1, a and b in the context of the main loop 
        # will be >= 1. LCM with 0 is typically defined as 0.
        if a == 0 or b == 0:
             return 0
             
        # Calculate gcd(a, b). Since a, b >= 1, their gcd will be >= 1.
        g = self.gcd(a, b)
        
        # Calculate lcm using the formula lcm(a, b) = (|a * b|) / gcd(a, b).
        # We use integer division //. abs() is not necessary as inputs are positive.
        # Python's arbitrary precision integers handle the potential large size of a * b.
        # The formula (a // g) * b could be used to mitigate intermediate overflow in 
        # languages with fixed-size integers, but it's not required here.
        res = (a * b) // g
        return res

    def maxLength(self, nums: List[int]) -> int:
        """
        Finds the length of the longest product equivalent subarray of nums.
        An array arr is called product equivalent if prod(arr) == lcm(arr) * gcd(arr).

        Args:
            nums: A list of positive integers.

        Returns:
            The length of the longest product equivalent subarray.
        """
        n = len(nums)
        # Constraints state: 2 <= nums.length <= 100, and 1 <= nums[i] <= 10.
        
        max_len = 0 # Initialize the maximum length found so far

        # Iterate through all possible starting indices 'i' of the subarrays.
        for i in range(n):
            # Initialize the product, GCD, and LCM for the subarray starting at index i.
            # These values will be updated as we extend the subarray to the right.
            current_prod = 1
            # We initialize current_gcd_val and current_lcm_val based on the first element (when j=i).
            # Using placeholder values here is okay as they get overwritten immediately.
            current_gcd_val = 0 
            current_lcm_val = 1 

            # Iterate through all possible ending indices 'j' for subarrays starting at 'i'.
            # This loop considers subarrays nums[i..j].
            for j in range(i, n):
                num = nums[j] # The current number being included in the subarray.
                
                # Update the product of elements in the current subarray nums[i..j].
                # Python's arbitrary precision integers handle potential large product values.
                current_prod *= num

                # Update the GCD and LCM of elements in the current subarray nums[i..j].
                if j == i:
                    # If this is the first element (subarray nums[i..i]), it defines the initial GCD and LCM.
                    current_gcd_val = num
                    current_lcm_val = num
                else:
                    # For subsequent elements, update GCD and LCM incrementally.
                    # The new GCD is gcd(previous_gcd, current_number).
                    current_gcd_val = self.gcd(current_gcd_val, num)
                    # The new LCM is lcm(previous_lcm, current_number).
                    current_lcm_val = self.lcm(current_lcm_val, num)
                    # The maximum possible LCM value is bounded (lcm(1..10) = 2520), 
                    # so performance of LCM calculation is not a concern.

                # Check if the current subarray nums[i..j] satisfies the product equivalent condition:
                # prod(arr) == lcm(arr) * gcd(arr)
                # Since all nums[i] >= 1, current_gcd_val >= 1 and current_lcm_val >= 1,
                # avoiding issues with multiplication by zero.
                if current_prod == current_lcm_val * current_gcd_val:
                    # If the condition holds, calculate the length of the current subarray.
                    current_len = j - i + 1
                    # Update the overall maximum length found so far.
                    max_len = max(max_len, current_len)
                        
        # After checking all possible subarrays, return the maximum length found.
        # It's known that any subarray of length 2 satisfies a*b = lcm(a,b)*gcd(a,b).
        # Since n >= 2, the function will always find a product equivalent subarray
        # of length at least 2 (unless all elements cause issues, which seems unlikely
        # with positive integers). If n=2, max_len >= 2. If n>2, max_len could be larger.
        # If nums only contains 1s, max_len will be n.
        return max_len