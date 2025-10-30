class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        # Find number of distinct elements in entire array
        total_distinct = len(set(nums))
        
        count = 0
        n = len(nums)
        
        # Check all possible subarrays
        for i in range(n):
            distinct_set = set()
            for j in range(i, n):
                distinct_set.add(nums[j])
                if len(distinct_set) == total_distinct:
                    # All subarrays from i to j, j+1, ..., n-1 are complete
                    count += n - j
                    break
        
        return count