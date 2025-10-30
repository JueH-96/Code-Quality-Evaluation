class Solution:
    def numberOfGoodPartitions(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        
        # Get last occurrence of each number
        last = {}
        for i, num in enumerate(nums):
            last[num] = i
            
        # Count number of non-overlapping intervals
        count = 1
        max_end = 0
        i = 0
        
        while i < len(nums):
            max_end = last[nums[i]]
            j = i + 1
            
            # Extend interval if there's overlap
            while j <= max_end:
                max_end = max(max_end, last[nums[j]])
                j += 1
                
            if max_end != len(nums)-1:
                count += 1
                
            i = j
            
        # Return 2^(count-1) since each partition point can be used or not
        return pow(2, count-1, MOD)