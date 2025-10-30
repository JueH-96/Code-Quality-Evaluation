import collections
import math
from typing import List

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # counts[k] stores the number of times 2^k appears from initial numbers
        # and intermediate splits that haven't been turned into carry yet.
        counts = collections.defaultdict(int)
        total_sum = 0
        for x in nums:
            # nums[i] >= 1 guarantees x > 0
            k = int(math.log2(x))
            counts[k] += 1
            total_sum += x

        # If the total sum of elements is less than target, it's impossible
        if total_sum < target:
            return -1

        ops = 0
        # carry represents the number of 2^k units carried over from processing the previous power (k-1).
        # Specifically, total_available_at_level // 2 from the previous iteration becomes carry for the current k.
        carry = 0

        # Iterate through powers of 2, from 2^0 upwards.
        # Target < 2^31, so bits up to 30 are relevant. Max initial power is 30.
        # Carry propagates upwards. Let's process up to k=32 to ensure all carries are handled.
        # This covers powers 2^0 through 2^32.
        max_k_to_process = 33 # Process powers 0 through 32

        for k in range(max_k_to_process):
            # total_available_at_level is the effective number of 2^k units we have at the start of processing level k.
            # It comes from the initial/split items at power k (counts[k]) plus the carry from power k-1.
            current_power_count = counts.get(k, 0)
            total_available_at_level = current_power_count + carry

            # Check if the k-th bit of target is set (i.e., we need one 2^k for the target sum)
            required_k = (target >> k) & 1

            if required_k == 1:
                # We need one 2^k unit for the target sum

                if total_available_at_level >= 1:
                    # We have enough units at this level. Use one unit.
                    total_available_at_level -= 1
                else:
                    # We don't have a 2^k unit from current_power_count or carry.
                    # We must obtain one by splitting a larger power 2^j (j > k).
                    # Find the smallest power j > k that we have at least one unit of in the counts map.
                    # Counts map contains items that haven't been converted to carry yet (original items or intermediate splits).
                    j_found = -1
                    
                    # Search for the smallest j > k that has a positive count in the counts map.
                    # Keys in counts map are <= 30 (from original nums) or intermediate splits <= 29.
                    # The search only needs to go up to max_k_to_process.
                    for j in range(k + 1, max_k_to_process):
                         if counts.get(j, 0) > 0:
                            j_found = j
                            break

                    # Based on the total_sum >= target check, a suitable j must exist if needed.
                    # If j_found is still -1 here, it implies an issue with the overall logic or constraints.
                    # assert j_found != -1 # In a contest, might need to handle this explicitly, but typically guaranteed.

                    # Assuming j_found is found:
                    counts[j_found] -= 1 # Use one unit of 2^j from counts
                    if counts[j_found] == 0:
                        del counts[j_found] # Clean up map entry if count becomes 0

                    # The operation 2^j -> 2 * 2^(j-1) -> ... -> 2 * 2^k takes j-k steps (operations).
                    ops += j_found - k

                    # This sequence of splits generates one 2^i for each i from k to j-1.
                    # We use one 2^k for the target. The items 2^(k+1) ... 2^(j-1) become available in counts.
                    # Add these newly available intermediate powers to counts.
                    for i in range(k + 1, j_found):
                        counts[i] = counts.get(i, 0) + 1

                    # total_available_at_level remains 0 in this branch because the required 2^k was obtained from a split and used immediately.

            # After considering the target bit (and potentially using one unit from total_available_at_level),
            # total_available_at_level holds the number of 2^k units remaining at this level.
            # These contribute to the carry for the next level (k+1).
            # Two units of 2^k combine to form one unit of 2^(k+1).
            # The carry is calculated from the remaining total_available_at_level units at this power level.
            carry = total_available_at_level // 2

        # After iterating through all relevant power bits, ops holds the minimum operations.
        # The total_sum check at the beginning ensures feasibility if the sum is sufficient.
        # The loop processes bits from 0 up to max_k_to_process, ensuring carries are propagated
        # and needed powers are sourced either from available units or by splitting larger ones.

        return ops