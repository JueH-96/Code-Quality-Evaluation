class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # Count frequency of each power of 2
        from collections import defaultdict
        count = defaultdict(int)
        total_sum = 0
        
        for num in nums:
            count[num] += 1
            total_sum += num
        
        # If total sum is less than target, impossible
        if total_sum < target:
            return -1
        
        operations = 0
        power = 1
        
        # Process each bit of target from least significant to most significant
        while target > 0:
            # If current bit is set in target
            if target & 1:
                # We need at least one coin of current power
                if count[power] > 0:
                    # Use one coin of current power
                    count[power] -= 1
                else:
                    # Need to break larger coins to get current power
                    # Find the smallest larger power that we have
                    larger_power = power * 2
                    while larger_power <= 2**30 and count[larger_power] == 0:
                        larger_power *= 2
                    
                    # If no larger power available, impossible
                    if larger_power > 2**30:
                        return -1
                    
                    # Break the larger coin down to current power
                    count[larger_power] -= 1
                    temp_power = larger_power
                    
                    # Keep breaking until we reach current power
                    while temp_power > power:
                        operations += 1
                        temp_power //= 2
                        count[temp_power] += 2
                    
                    # Now use one coin of current power
                    count[power] -= 1
            
            # Move to next bit
            target >>= 1
            power *= 2
        
        return operations