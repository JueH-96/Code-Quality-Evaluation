class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        # Count distinct elements in the entire array
        total_distinct = len(set(nums))
        
        count = 0
        n = len(nums)
        
        # Check all possible subarrays
        for i in range(n):
            distinct_elements = set()
            for j in range(i, n):
                distinct_elements.add(nums[j])
                
                # If current subarray has same number of distinct elements
                # as the entire array, it's complete
                if len(distinct_elements) == total_distinct:
                    count += 1
                # If we already have more distinct elements than possible,
                # no need to continue (this won't happen given constraints)
                elif len(distinct_elements) > total_distinct:
                    break
        
        return count