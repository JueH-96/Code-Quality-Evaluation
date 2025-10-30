import collections

class Solution:
    def minGroupsForValidAssignment(self, nums: List[int]) -> int:
        counts = collections.Counter(nums)
        value_counts = list(counts.values())
        max_count = 0
        if value_counts:
            max_count = max(value_counts)
        min_groups = float('inf')
        if not value_counts:
            return 0
        
        for s in range(1, max_count + 1):
            current_groups = 0
            possible_for_s = True
            for count in value_counts:
                if count < s:
                    possible_for_s = False
                    break
                num_groups_s_plus_1 = count // (s + 1)
                remainder = count % (s + 1)
                num_groups_s = 0
                if remainder > 0:
                    num_groups_s = 1
                current_groups += num_groups_s_plus_1 + num_groups_s
            if possible_for_s:
                min_groups = min(min_groups, current_groups)
                
        if min_groups == float('inf'):
            # try base size 1. sizes 1, 2. 
            s = 1
            current_groups = 0
            possible_for_s = True
            for count in value_counts:
                if count < s:
                    possible_for_s = False
                    break
                num_groups_s_plus_1 = count // (s + 1)
                remainder = count % (s + 1)
                num_groups_s = 0
                if remainder > 0:
                    num_groups_s = 1
                current_groups += num_groups_s_plus_1 + num_groups_s
            if possible_for_s:
                min_groups = min(min_groups, current_groups)

        if min_groups == float('inf'):
            # If even s=1 doesn't work, something is wrong. But it should always work for s=1 if counts are >= 1. 
            # Let's reconsider the range for s. 
            # We need to find a group size s such that for every count c_v, we can partition it into groups of size s or s+1. 
            # We tried s from 1 to max_count. 
            pass

        if min_groups == float('inf'):
            # In case no s in range 1 to max_count worked, try s=1 again, as it should always work.
            s = 1
            current_groups = 0
            for count in value_counts:
                num_groups_s_plus_1 = count // (s + 1)
                remainder = count % (s + 1)
                num_groups_s = 0
                if remainder > 0:
                    num_groups_s = 1
                current_groups += num_groups_s_plus_1 + num_groups_s
            min_groups = min(min_groups, current_groups)

        return min_groups