class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        # Get the number of distinct elements in the entire array
        distinct_count = len(set(nums))
        n = len(nums)
        count = 0
        
        # For each possible starting position
        for left in range(n):
            # Keep track of distinct elements seen in current subarray
            distinct_set = set()
            # Expand the subarray to the right
            for right in range(left, n):
                distinct_set.add(nums[right])
                # Once we have all distinct elements
                if len(distinct_set) == distinct_count:
                    # All further extensions of this subarray will also be complete
                    count += n - right
                    break
        
        return count