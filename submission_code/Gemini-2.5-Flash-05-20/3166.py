import collections
import math # Not strictly necessary if using integer division carefully, but good for clarity of concept

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        # Step 1: Count frequencies of each number in nums.
        counts = collections.Counter(nums)
        
        # Step 2: Get a list of just the distinct frequencies.
        distinct_counts = list(counts.values())
        
        # Step 3: Determine the maximum possible value for 's', the smaller group size.
        # 's' cannot be greater than the minimum frequency of any number,
        # because if it were, a number with that minimum frequency couldn't form a group.
        min_overall_count = min(distinct_counts)
        
        # Initialize min_total_groups to a very large number.
        min_total_groups = float('inf')
        
        # Step 4: Iterate 's' from 1 up to min_overall_count.
        # 's' represents the smaller of the two allowed group sizes (s, s+1).
        for s in range(1, min_overall_count + 1):
            is_s_valid = True
            current_total_groups = 0
            
            # Step 5: For the current 's', check if it's valid for all distinct counts.
            for c in distinct_counts:
                # We need to distribute 'c' items into groups of size 's' or 's+1'.
                # To minimize the number of groups for this 'c', we prioritize using 's+1' groups.
                num_s_plus_1_groups = c // (s + 1)
                
                # The remaining elements must form groups of size 's'.
                remaining_for_s_groups = c % (s + 1)
                
                # If the remaining elements cannot be perfectly divided into groups of size 's',
                # then this 's' is not a valid minimum group size for this 'c' (and thus not for the whole assignment).
                if remaining_for_s_groups % s != 0:
                    is_s_valid = False
                    break # This 's' value is not suitable, move to the next 's'.
                
                # If valid, calculate the number of groups of size 's'.
                num_s_groups = remaining_for_s_groups // s
                
                # Add the groups formed for this count 'c' to the running total for the current 's'.
                current_total_groups += (num_s_plus_1_groups + num_s_groups)
            
            # Step 5c: If the current 's' was valid for all distinct counts,
            # update the overall minimum total groups found so far.
            if is_s_valid:
                min_total_groups = min(min_total_groups, current_total_groups)
                
        # Step 6: Return the minimum total groups found.
        return min_total_groups