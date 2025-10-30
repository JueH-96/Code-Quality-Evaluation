class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        for x in nums:
            if x < k:
                return -1
        
        operations = 0
        current_nums = list(nums)
        
        while True:
            greater_than_k_values = set()
            for x in current_nums:
                if x > k:
                    greater_than_k_values.add(x)
            
            if not greater_than_k_values:
                all_equal_k = True
                for x in current_nums:
                    if x != k:
                        all_equal_k = False
                        break
                if all_equal_k:
                    return operations
                else:
                    return -1 # Should not reach here based on problem description and initial check.
                    
            sorted_greater_values = sorted(list(greater_than_k_values), reverse=True)
            target_value = sorted_greater_values[0]
            next_value = k
            if len(sorted_greater_values) > 1:
                next_value = sorted_greater_values[1]
                
            h = next_value
            
            is_h_valid = True
            greater_than_h_elements = []
            for x in current_nums:
                if x > h:
                    greater_than_h_elements.append(x)
                    
            if greater_than_h_elements:
                first_val = greater_than_h_elements[0]
                for val in greater_than_h_elements:
                    if val != first_val:
                        is_h_valid = False
                        break
            else:
                is_h_valid = True
                
            if is_h_valid:
                next_nums = []
                operation_performed = False
                for x in current_nums:
                    if x > h:
                        next_nums.append(h)
                        operation_performed = True
                    else:
                        next_nums.append(x)
                if operation_performed:
                    operations += 1
                    current_nums = next_nums
                else:
                    # No operation performed in this iteration, but still elements > k exist? 
                    # This should not happen if we are picking h correctly.
                    # Let's try to use target_value - 1 as h if next_value == k
                    if next_value == k and target_value > k:
                        h = target_value - 1
                        is_h_valid_alt = True
                        greater_than_h_elements_alt = []
                        for x in current_nums:
                            if x > h:
                                greater_than_h_elements_alt.append(x)
                        if greater_than_h_elements_alt:
                            first_val_alt = greater_than_h_elements_alt[0]
                            for val_alt in greater_than_h_elements_alt:
                                if val_alt != first_val_alt:
                                    is_h_valid_alt = False
                                    break
                        else:
                            is_h_valid_alt = True
                            
                        if is_h_valid_alt:
                            next_nums_alt = []
                            operation_performed_alt = False
                            for x in current_nums:
                                if x > h:
                                    next_nums_alt.append(h)
                                    operation_performed_alt = True
                                else:
                                    next_nums_alt.append(x)
                            if operation_performed_alt:
                                operations += 1
                                current_nums = next_nums_alt
                                continue # restart while loop
                                
                        else:
                            return -1 # Should not reach here based on algorithm design
                    else:
                        return -1 # Should not reach here based on algorithm design
                        
            else:
                return -1 # Should not reach here based on algorithm design
                
            all_k = True
            for x in current_nums:
                if x != k:
                    all_k = False
                    break
            if all_k:
                return operations