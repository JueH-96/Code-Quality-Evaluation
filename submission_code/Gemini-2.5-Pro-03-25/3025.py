import math
from collections import Counter
from typing import List

class Solution:
    """
    Solves the problem of finding the minimum operations to form a subsequence sum equal to target.
    An operation involves replacing a power of 2, x > 1, with two copies of x/2.
    The approach uses a greedy strategy based on the binary representation of the target, processing from least significant bit (LSB) upwards.
    """
    def minOperations(self, nums: List[int], target: int) -> int:
        
        # Calculate the total sum of elements in nums.
        # If the total sum is less than the target, it's impossible to form
        # a subsequence summing to target, as operations preserve the total sum.
        current_sum = sum(nums)
        if current_sum < target:
            return -1
        
        # Use a Counter (frequency map) to store the count of each power of 2 present in nums.
        # The keys of the counter will be the exponents p, where an element is 2^p.
        counts = Counter()
        for x in nums:
            # Process only positive powers of 2. Constraints state non-negative powers of 2.
            # 2^0 = 1 is non-negative. We assume nums contains positive integers based on examples and constraints.
            if x > 0: 
                # Calculate the exponent p such that x = 2^p.
                # Using x.bit_length() - 1 is an efficient way to find the exponent for powers of 2.
                power = x.bit_length() - 1
                counts[power] += 1
            
        ops = 0  # Initialize the count of operations
        
        # Process powers of 2 from 2^0 upwards. We need to go high enough to cover the target's
        # bits and any potential carry propagation.
        # Max element in nums is 2^30. Max target < 2^31. Max sum can reach ~2^40.
        # A loop limit up to 60 (covering powers up to 2^60) is sufficiently large to handle all cases.
        max_power_idx = 61 # Defines the upper bound for powers 2^i we consider.

        for i in range(max_power_idx):
            
            # Check if the i-th bit is set in the target. This tells us if we need a 2^i component for the target sum.
            # (target >> i) & 1 extracts the i-th bit.
            needed = (target >> i) & 1
            
            # If the i-th bit is needed (equals 1)
            if needed == 1:
                if counts[i] > 0:
                    # If we have one or more 2^i available, use one.
                    counts[i] -= 1
                else:
                    # If we need 2^i but have none available (counts[i] == 0).
                    # We must generate one by breaking down a larger power of 2.
                    # Find the smallest available power 2^k such that k > i.
                    k = i + 1
                    while k < max_power_idx and counts[k] == 0:
                        k += 1
                    
                    # If no larger power is found within our processing range, it implies impossibility.
                    # This theoretically shouldn't happen if the initial sum check passed,
                    # as it would mean the total available sum from powers >= 2^i is zero,
                    # while the target still requires a component >= 2^i.
                    if k == max_power_idx:
                         return -1 
                    
                    # Found the smallest k. Perform operations to break down 2^k.
                    # Breaking 2^k down to generate 2^i requires k - i operations.
                    # Example: 8 (2^3) -> 4,4 (1 op); 4 -> 2,2 (1 op); 2 -> 1,1 (1 op). Total 3 = 3-0 ops.
                    ops += (k - i)
                    
                    # Update the counts based on the breakdown process:
                    # One 2^k is consumed.
                    counts[k] -= 1
                    # The breakdown process $2^k \to 2 \times 2^{k-1} \to \dots \to 2 \times 2^i$ leaves behind
                    # one instance of each intermediate power $2^j$ for $j$ from $i+1$ to $k-1$.
                    for j in range(i + 1, k):
                        counts[j] += 1
                    # The final step $2^{i+1} \to 2 \times 2^i$ generates two instances of $2^i$.
                    counts[i] += 2
                    
                    # Now we have at least two 2^i available. Use one for the target requirement.
                    counts[i] -= 1

            # After potentially using one 2^i for the target, handle the carry-over.
            # Any remaining pairs of 2^i can be combined to form one 2^{i+1}.
            carry = counts[i] // 2
            if carry > 0:
                # Add the carry to the count of the next power level 2^{i+1}.
                # Check bounds to avoid accessing counts beyond max_power_idx implicitly.
                if i + 1 < max_power_idx:
                   counts[i+1] += carry
            
            # The count of 2^i effectively becomes counts[i] % 2 after carry calculation,
            # but we don't need to explicitly update counts[i] as it's not read in subsequent iterations
            # except via the carry mechanism to counts[i+1].

        # If the loop completes without returning -1, it means we could satisfy all bits of the target.
        # The accumulated 'ops' is the minimum number of operations.
        return ops