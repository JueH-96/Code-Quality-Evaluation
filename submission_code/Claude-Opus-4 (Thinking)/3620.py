class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        nums.sort()
        count = 0
        last_assigned = float('-inf')
        
        for num in nums:
            # The value we can assign is in [num - k, num + k]
            # We want the smallest value > last_assigned
            min_val = max(num - k, last_assigned + 1)
            
            if min_val <= num + k:
                last_assigned = min_val
                count += 1
        
        return count