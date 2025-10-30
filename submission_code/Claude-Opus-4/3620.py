class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        nums.sort()
        
        # Keep track of the last position we used
        last_used = float('-inf')
        distinct_count = 0
        
        for num in nums:
            # The range where we can place this element is [num-k, num+k]
            # We want to place it at the smallest available position > last_used
            min_pos = num - k
            max_pos = num + k
            
            # Find the smallest position we can use
            if min_pos > last_used:
                # We can use min_pos
                last_used = min_pos
                distinct_count += 1
            elif max_pos > last_used:
                # We can use last_used + 1 (if it's within range)
                last_used = last_used + 1
                distinct_count += 1
            # else: we can't place this element without creating a duplicate
        
        return distinct_count