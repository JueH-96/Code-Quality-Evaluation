from typing import List

class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        if any(num < k for num in nums):
            return -1
        operations = 0
        while True:
            greater_than_k_values = [num for num in nums if num > k]
            if not greater_than_k_values:
                break
            unique_greater_than_k = sorted(list(set(greater_than_k_values)), reverse=True)
            if not unique_greater_than_k:
                break
            max_val_greater_than_k = unique_greater_than_k[0]
            
            target_h = -1
            unique_nums_desc = sorted(list(set(nums)), reverse=True)
            
            found_next_h = False
            for unique_val in unique_nums_desc:
                if unique_val < max_val_greater_than_k and unique_val >= k:
                    target_h = unique_val
                    found_next_h = True
                    break
            if not found_next_h:
                target_h = k
                
            is_valid_h = True
            values_greater_than_h = []
            for num in nums:
                if num > target_h:
                    values_greater_than_h.append(num)
            if values_greater_than_h:
                first_val = values_greater_than_h[0]
                for val in values_greater_than_h:
                    if val != first_val:
                        is_valid_h = False
                        break
                        
            if is_valid_h:
                next_nums = []
                for num in nums:
                    if num > target_h:
                        next_nums.append(target_h)
                    else:
                        next_nums.append(num)
                nums = next_nums
                operations += 1
            else:
                # This case should not happen based on our logic, but for robustness.
                return -1 
                
        all_equal_k = True
        for num in nums:
            if num != k:
                all_equal_k = False
                break
        if all_equal_k:
            return operations
        else:
            return -1