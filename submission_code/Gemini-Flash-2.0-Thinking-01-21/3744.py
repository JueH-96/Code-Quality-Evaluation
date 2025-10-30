from typing import List

class Solution:
    def minOperations(self, queries: List[List[int]]) -> int:

        def S(N):
            """
            Calculates the sum of get_steps(x) for x in the range [1, N].
            get_steps(x) is the minimum number of times floor(./4) must be applied
            to x to reach 0. For x > 0, this is the smallest k >= 1 such that 4^k > x.
            Equivalently, get_steps(x) = k iff 4^(k-1) <= x <= 4^k - 1 for k >= 1.
            """
            if N == 0:
                return 0
            
            total_sum = 0
            k_power_index = 0 # Represents k in 4^k, corresponds to step_count = k_power_index + 1
            
            # The range of numbers 'x' such that get_steps(x) = k_power_index + 1
            # is [4^k_power_index, 4^(k_power_index + 1) - 1].
            # Let current_lower_bound = 4^k_power_index.
            current_lower_bound = 1 # For k_power_index = 0, 4^0 = 1

            # Iterate through levels defined by powers of 4.
            # The loop continues as long as the current level's range starts within [1, N].
            while current_lower_bound <= N:
                step_count = k_power_index + 1
                lower_bound_for_level = current_lower_bound # This is 4^k_power_index
                
                # Calculate the upper bound for this level: 4^(k_power_index + 1) - 1.
                # This is equivalent to 4 * (4^k_power_index) - 1 = 4 * lower_bound_for_level - 1.
                # Python's arbitrary precision integers handle large values, so no overflow concerns here
                # for values within typical problem constraints like N <= 10^9.
                upper_bound_for_level = 4 * lower_bound_for_level - 1
                
                # The numbers in [1, N] that require step_count steps are in the intersection
                # of [1, N] and [lower_bound_for_level, upper_bound_for_level].
                # Since lower_bound_for_level starts at 1 and increases, the intersection is
                # [lower_bound_for_level, min(N, upper_bound_for_level)].
                
                effective_upper_bound = min(N, upper_bound_for_level)
                
                # Count of numbers in this effective range.
                # Note: effective_upper_bound >= lower_bound_for_level because lower_bound_for_level <= N
                # and lower_bound_for_level <= upper_bound_for_level (for k_power_index >= 0).
                count = effective_upper_bound - lower_bound_for_level + 1
                
                # Each of these 'count' numbers requires 'step_count' operations individually.
                # Add their contribution to the total sum of individual steps.
                total_sum += step_count * count
                
                # Prepare for the next iteration.
                # The next level's range starts at 4^(k_power_index + 1), which is 4 * current_lower_bound.
                
                # Check if the next lower bound calculation might overflow if not using arbitrary precision
                # or if the next lower bound will certainly exceed N.
                # For N <= 10^9, current_lower_bound can go up to 4^15 (approx 10^9).
                # The next lower bound would be 4^16 (approx 4.3 * 10^9). This fits in a 64-bit integer.
                # The loop condition `current_lower_bound <= N` ensures we stop correctly.
                
                current_lower_bound = 4 * lower_bound_for_level
                k_power_index += 1

            return total_sum

        total_query_operations = 0
        for l, r in queries:
            # The initial array for query [l, r] is nums = [l, l+1, ..., r].
            # The minimum number of operations to reduce all elements in nums to zero
            # is ceil(Sum(get_steps(x) for x in nums) / 2).
            # This is because each operation (on two positive numbers) reduces the
            # total sum of get_steps values by 2.
            # Sum(get_steps(x) for x in range(l, r+1)) = sum(get_steps(x) for x in range(1, r+1)) - sum(get_steps(x) for x in range(1, l))
            # This is S(r) - S(l-1).

            sum_steps_in_range = S(r) - S(l - 1)
            
            # Minimum operations = ceil(sum_steps_in_range / 2).
            # Using integer division, ceil(a/b) is (a + b - 1) // b.
            # For b=2, this is (sum_steps_in_range + 2 - 1) // 2 = (sum_steps_in_range + 1) // 2.
            ops_for_query = (sum_steps_in_range + 1) // 2
            
            total_query_operations += ops_for_query
            
        return total_query_operations