class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        # Count the total number of distinct elements in the array
        total_distinct = len(set(nums))
        n = len(nums)
        
        result = 0
        
        # For each starting index i, move j until we have a subarray [i..j]
        # containing all distinct elements. Once found, all subarrays extending
        # beyond j to the right also include all distinct elements, so we can
        # count them in one step as (n - j).
        for i in range(n):
            freq = {}
            distinct_count = 0
            
            for j in range(i, n):
                freq[nums[j]] = freq.get(nums[j], 0) + 1
                if freq[nums[j]] == 1:
                    distinct_count += 1
                    
                # If we've reached a subarray containing all distinct elements,
                # every subarray starting at i and ending in [j..n-1] is complete.
                if distinct_count == total_distinct:
                    result += (n - j)
                    break
        
        return result