from typing import List

class Solution:
    def maximumSubarraySum(self, nums: List[int], k: int) -> int:
        current_pre = 0
        hash_map = {}
        max_sum = -float('inf')
        
        for j in range(len(nums)):
            current_val = nums[j]
            pre_j_plus_1 = current_pre + current_val
            
            # Collect all candidates
            candidates = []
            val1 = current_val - k
            if val1 in hash_map:
                candidates.append(pre_j_plus_1 - hash_map[val1])
            val2 = current_val + k
            if val2 in hash_map:
                candidates.append(pre_j_plus_1 - hash_map[val2])
            
            # Update max_sum if there are any valid candidates
            if candidates:
                current_max = max(candidates)
                if current_max > max_sum:
                    max_sum = current_max
            
            # Update the hash_map for current_val
            if current_val not in hash_map:
                hash_map[current_val] = current_pre
            else:
                if current_pre < hash_map[current_val]:
                    hash_map[current_val] = current_pre
            
            # Move to next prefix sum
            current_pre = pre_j_plus_1
        
        return max_sum if max_sum != -float('inf') else 0