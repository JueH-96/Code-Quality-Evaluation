class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        # Count distinct elements in the entire array
        total_distinct = len(set(nums))
        
        n = len(nums)
        count = 0
        
        # For each starting position
        for i in range(n):
            # Track distinct elements in current subarray
            distinct_elements = set()
            
            # Extend the subarray from position i
            for j in range(i, n):
                distinct_elements.add(nums[j])
                
                # If this subarray is complete
                if len(distinct_elements) == total_distinct:
                    # All subarrays from [i, j] to [i, n-1] are complete
                    count += n - j
                    break
        
        return count