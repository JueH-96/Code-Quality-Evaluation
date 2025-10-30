class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        # Sort the array
        nums.sort()
        
        distinct_count = 0
        last_used = -10**18  # a very small sentinel value
        
        for x in nums:
            # possible range to place x is [x-k, x+k]
            low = x - k
            high = x + k
            
            # we want the placed value to be at least last_used + 1 to keep them distinct
            desired = max(low, last_used + 1)
            
            # if we can place x in a valid spot within [low, high]
            if desired <= high:
                distinct_count += 1
                last_used = desired  # update the last used position
        
        return distinct_count