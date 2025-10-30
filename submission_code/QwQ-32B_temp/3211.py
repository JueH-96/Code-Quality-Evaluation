import bisect

class Solution:
    def findMaximumLength(self, nums: List[int]) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + nums[i]
        
        candidates = []
        v_list = []
        
        # Initialize for the first element (j=0)
        first_v = 2 * prefix[1]
        candidates.append( (first_v, 1, prefix[1], prefix[1]) )
        v_list.append(first_v)
        current_max = 1
        
        for i in range(1, n):
            current_prefix = prefix[i+1]
            target = current_prefix
            
            # Find the largest v_j <= target using bisect
            idx = bisect.bisect_right(v_list, target) - 1
            best_j = idx
            
            if best_j >= 0:
                # Found a valid j
                v_j, dp_j, prefix_j, s_j = candidates[best_j]
                new_dp = dp_j + 1
                new_s = current_prefix - prefix_j
                new_v = current_prefix + new_s
            else:
                # Must merge with previous segment (last entry)
                v_j, dp_j, prefix_j, s_j = candidates[-1]
                new_s = s_j + (current_prefix - prefix_j)
                new_v = current_prefix + new_s
                new_dp = dp_j
            
            new_entry = (new_v, new_dp, current_prefix, new_s)
            
            # Maintain the candidates list
            # Remove entries from the end where v >= new_v and dp <= new_dp
            while candidates and (candidates[-1][0] >= new_v and candidates[-1][1] <= new_dp):
                candidates.pop()
                v_list.pop()
            
            # Check if we can add the new entry
            if not candidates or candidates[-1][0] < new_v:
                candidates.append(new_entry)
                v_list.append(new_v)
            
            # Update the current_max
            if new_dp > current_max:
                current_max = new_dp
        
        return current_max