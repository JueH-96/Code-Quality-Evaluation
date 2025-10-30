class Solution:
    def countWays(self, nums: List[int]) -> int:
        n = len(nums)
        count_ways = 0
        for k in range(n + 1):
            is_possible_k = True
            for x in nums:
                if x == k:
                    is_possible_k = False
                    break
            if not is_possible_k:
                continue
            selected_count = 0
            for x in nums:
                if x < k:
                    selected_count += 1
            if selected_count == k:
                count_ways += 1
        
        is_k0_valid = True
        for x in nums:
            if not (0 < x):
                is_k0_valid = False
                break
        if is_k0_valid:
            count_ways += 1
        elif k == 0 and all(num > 0 for num in nums): # Special case for k=0, if all nums > 0
            count_ways_k0 = 0
            possible_k0 = True
            for i in range(n):
                if not (0 < nums[i]):
                    possible_k0 = False
                    break
            if possible_k0:
                count_ways_k0 += 1
            
        # Re-evaluate k=0 condition.
        valid_k0 = True
        for num in nums:
            if not (0 < num):
                valid_k0 = False
                break
        if valid_k0:
            count_ways_k0_check = 1
        else:
            count_ways_k0_check = 0

        final_count_ways = 0
        for k in range(n + 1):
            is_possible_k = True
            for x in nums:
                if x == k:
                    is_possible_k = False
                    break
            if not is_possible_k:
                continue
            selected_count = 0
            for x in nums:
                if x < k:
                    selected_count += 1
            if selected_count == k:
                final_count_ways += 1
        
        is_k0_valid_final = True
        for num in nums:
            if not (0 < num):
                is_k0_valid_final = False
                break
        if is_k0_valid_final:
            final_count_ways += 1
            
        return final_count_ways