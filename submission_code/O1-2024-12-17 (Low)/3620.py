class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        # Sort the array so we can greedily place each element
        nums.sort()
        
        # We'll keep track of the last placed value in "placed"
        # Start it as a very negative number, so the first element is freely placeable
        placed = -10**20
        distinct_count = 0
        
        for x in nums:
            # We aim to place x in the smallest valid position >= placed+1 
            # within [x-k, x+k]
            candidate = max(placed + 1, x - k)
            if candidate <= x + k:
                # We can place x here
                distinct_count += 1
                placed = candidate
            # Otherwise, we skip x if there's no valid spot
                
        return distinct_count