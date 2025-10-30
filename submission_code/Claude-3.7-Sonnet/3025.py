class Solution:
    def minOperations(self, nums: List[int], target: int) -> int:
        # If sum of nums < target, it's impossible
        if sum(nums) < target:
            return -1
        
        # Count occurrences of each power of 2
        count = [0] * 31
        for num in nums:
            for i in range(31):
                if num == (1 << i):
                    count[i] += 1
                    break
        
        operations = 0
        
        for i in range(31):
            # If this bit is set in the target
            if target & (1 << i):
                if count[i] > 0:
                    count[i] -= 1
                else:
                    # Find the smallest larger power that has count > 0
                    j = i + 1
                    while j < 31 and count[j] == 0:
                        j += 1
                    
                    if j == 31:
                        return -1  # Can't form the target
                    
                    # Break down the larger power
                    operations += j - i
                    count[j] -= 1
                    
                    # For each level of breakdown, we get extra powers
                    for k in range(i, j):
                        count[k] += 1
                    
                    count[i] -= 1  # Use one for the target
            
            # Carry excess to the next position
            if i < 30:
                count[i + 1] += count[i] // 2
                count[i] %= 2
        
        return operations