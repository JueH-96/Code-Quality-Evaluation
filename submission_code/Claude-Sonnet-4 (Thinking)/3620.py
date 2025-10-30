class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        nums.sort()
        last_assigned = -2 * 10**9  # Initialize to a very small value
        count = 0
        
        for num in nums:
            min_val = num - k
            max_val = num + k
            
            # Try to assign the smallest possible value > last_assigned
            next_val = max(min_val, last_assigned + 1)
            
            if next_val <= max_val:
                last_assigned = next_val
                count += 1
        
        return count