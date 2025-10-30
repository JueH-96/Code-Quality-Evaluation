import collections
from typing import List

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        """
        Calculates the minimum number of operations to form a subsequence summing to target.
        """
        # Step 1: Check if the total sum is sufficient. If not, it's impossible.
        if sum(nums) < target:
            return -1

        # Step 2: Count the frequency of each power of two.
        # power_counts[k] will store the number of times 2^k appears in nums.
        power_counts = collections.Counter()
        for x in nums:
            power_counts[x.bit_length() - 1] += 1

        ops = 0
        current_sum = 0
        
        # We need to iterate up to a sufficiently large power.
        # Max power in nums is 30. Max in target is 30.
        # Sum of nums can be up to 1000 * 2^30 ~= 2^39.96.
        # A limit of 45 is safe to cover all possible powers we might need to access.
        limit = 45 

        # Step 3: Iterate through bits from LSB to MSB.
        for i in range(limit):
            # Add the value of all 2^i numbers from the initial list to our running sum.
            current_sum += power_counts[i] * (1 << i)

            # Check if the i-th bit of target is set, meaning we need a 2^i.
            if (target >> i) & 1:
                needed = 1 << i
                if current_sum >= needed:
                    # We have enough value from smaller powers to form a 2^i.
                    current_sum -= needed
                else:
                    # Not enough value. We must break down a larger power.
                    # Greedily find the smallest available power j > i.
                    j = i + 1
                    while j < limit and power_counts[j] == 0:
                        j += 1
                    
                    # Such a `j` must exist because we passed the initial sum check.
                    
                    # The cost to get a 2^i from a 2^j is j - i operations.
                    ops += j - i
                    # We use up one 2^j.
                    power_counts[j] -= 1
                    
                    # We effectively add the value of the broken-down number to our pool.
                    current_sum += (1 << j)
                    # Now, we can satisfy the need for 2^i.
                    current_sum -= needed
        
        return ops