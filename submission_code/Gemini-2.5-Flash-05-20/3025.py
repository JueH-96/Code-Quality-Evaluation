import collections
import math

class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # powers[k] stores the count of 2^k elements
        # Max power is 2^30. Target < 2^31 means max bit is 2^30.
        # Array size 32 is sufficient for indices 0 to 31.
        powers = [0] * 32 
        
        total_initial_sum = 0
        for num in nums:
            # log2(num) is the exponent k for 2^k.
            power_idx = int(math.log2(num))
            powers[power_idx] += 1
            total_initial_sum += num

        if total_initial_sum < target:
            return -1
        
        ops = 0
        # current_sum_available_as_power_k: represents the total value from
        # powers up to k that hasn't been consumed by target bits yet.
        # This value can be used to satisfy future target bits for higher k.
        current_sum_available_as_power_k = 0

        # Iterate from smallest power (2^0) up to the maximum needed (2^30).
        # Loop up to k=31 to process potential carries from 2^30.
        for k in range(32):
            if target == 0:
                break
            
            # Add the actual value of 2^k elements to the running sum.
            current_sum_available_as_power_k += powers[k] * (1 << k)
            
            # Check if the k-th bit of target is set (i.e., we need a 2^k).
            if (target >> k) & 1:
                if current_sum_available_as_power_k >= (1 << k):
                    # We have enough value accumulated from powers <= 2^k
                    # to satisfy the 2^k requirement.
                    current_sum_available_as_power_k -= (1 << k)
                else:
                    # We don't have enough value at or below 2^k to satisfy the requirement.
                    # We MUST break down a larger power.
                    
                    # Find the smallest power 2^j (where j > k) that we have.
                    found_j = -1
                    for j in range(k + 1, 32):
                        if powers[j] > 0:
                            found_j = j
                            break
                    
                    if found_j == -1:
                        # This means we cannot satisfy the target, even with breaking larger numbers.
                        # This check is important as total_initial_sum >= target doesn't guarantee forming specific bits.
                        return -1
                    
                    # The cost to get a 2^k from a 2^j is j-k operations.
                    # Example: 2^3 (8) to get 2^1 (2):
                    # 8 -> 4,4 (1 op)
                    # one 4 -> 2,2 (1 op)
                    # Total 2 ops (which is 3-1).
                    ops += (found_j - k)
                    
                    # Conceptually, this 2^j is used up to provide the 2^k.
                    # The problem wording is tricky, but this interpretation matches the examples.
                    # We don't need to explicitly update powers[found_j - 1] += 2 etc.,
                    # because we're thinking in terms of total 'value' that propagates.
                    powers[found_j] -= 1 # This 2^j is consumed.
                    
                    # The value 2^j has now been "broken down" and can contribute to current_sum_available_as_power_k.
                    current_sum_available_as_power_k += (1 << found_j)
                    # From this, we take out the 2^k we needed.
                    current_sum_available_as_power_k -= (1 << k)
            
            # After processing the current k-th bit, any remaining value in
            # current_sum_available_as_power_k can be conceptually carried over
            # to the (k+1)-th power. We effectively divide the sum by 2.
            current_sum_available_as_power_k //= 2
            
        return ops