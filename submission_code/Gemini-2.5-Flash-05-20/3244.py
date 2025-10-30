import math
from typing import List

class Solution:
    def minimumArrayLength(self, nums: List[int]) -> int:
        # Step 1: Calculate the GCD of all elements in nums.
        # Initialize current_gcd with the first element.
        # Iterate through the rest of the elements, updating current_gcd.
        current_gcd = nums[0]
        for x in nums[1:]:
            current_gcd = math.gcd(current_gcd, x)
        
        # Step 2: Find the minimum value in nums.
        min_val = min(nums)
        
        # Step 3: Apply the derived rule to determine the minimum length.
        # The minimum length is 2 if and only if all three conditions are met:
        # 1. The overall GCD of the array (current_gcd) is greater than 1.
        # 2. The minimum value in the initial array (min_val) is equal to the overall GCD (current_gcd).
        # 3. The initial length of the array (N) is 3 or more.
        # In all other cases, the minimum length is 1.

        # Condition 1: current_gcd > 1
        if current_gcd > 1:
            # Condition 2: min_val == current_gcd
            if min_val == current_gcd:
                # Condition 3: len(nums) >= 3
                if len(nums) >= 3:
                    return 2
                else:
                    # If len(nums) is 1 or 2, and min_val == current_gcd > 1
                    #   - If len(nums) == 1, array is [g], length 1.
                    #   - If len(nums) == 2, array is [g, g], perform g % g = 0, array becomes [0], length 1.
                    return 1
            else:
                # min_val != current_gcd (and current_gcd > 1)
                # In this case, we can always reduce the array to a single value (current_gcd) or 0.
                # Example: [6, 10, 15] -> min_val=6, current_gcd=1. (This would not enter this branch)
                # Example: Let's imagine if current_gcd was 2, and min_val was 4, like [4, 6, 10].
                # Here, min_val (4) != current_gcd (2). We can reduce 6%4=2, then have [4,2,10]. Then 4%2=0, then 10%2=0. Eventually [2]. Length 1.
                return 1
        else: # current_gcd == 1
            # If the GCD is 1, it means we can always obtain the value 1,
            # and then use 1 to reduce other elements to 0 (X % 1 = 0), finally leaving a single 0.
            # This leads to a length of 1.
            return 1