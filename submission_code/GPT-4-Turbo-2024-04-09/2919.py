class Solution:
    def maxIncreasingGroups(self, usageLimits: List[int]) -> int:
        # Sort the usageLimits with their indices to prioritize using smaller limits first
        indexed_limits = sorted((limit, i) for i, limit in enumerate(usageLimits))
        
        # This will track the remaining usage of each index
        remaining_usage = [0] * len(usageLimits)
        for limit, index in indexed_limits:
            remaining_usage[index] = limit
        
        # Start forming groups
        current_group_size = 0
        num_groups = 0
        
        while True:
            next_group_size = current_group_size + 1
            count = 0
            
            # Try to form the next group with size greater than the last group
            for i in range(len(remaining_usage)):
                if remaining_usage[i] > 0:
                    remaining_usage[i] -= 1
                    count += 1
                    if count == next_group_size:
                        break
            
            # If we successfully formed a group of the required size, increment the group count
            if count == next_group_size:
                num_groups += 1
                current_group_size = next_group_size
            else:
                # Restore the usage since we couldn't form a full group
                for j in range(i, i - count, -1):
                    remaining_usage[j] += 1
                break
        
        return num_groups