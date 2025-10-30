class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        # Find the total number of distinct elements in the array
        total_distinct = len(set(nums))
        
        count = 0
        n = len(nums)
        
        # Check all possible subarrays starting from each position
        for i in range(n):
            # Use a set to track distinct elements in current subarray
            distinct_in_subarray = set()
            
            for j in range(i, n):
                distinct_in_subarray.add(nums[j])
                
                # If the subarray has all distinct elements
                if len(distinct_in_subarray) == total_distinct:
                    # All subarrays from [i,j] to [i,n-1] are complete
                    count += n - j
                    break
        
        return count