class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        from collections import Counter
        
        # Get the number of distinct elements in the whole array
        total_distinct = len(set(nums))
        
        n = len(nums)
        count = 0
        
        # Iterate over all possible subarrays
        for i in range(n):
            distinct_count = 0
            freq = Counter()
            for j in range(i, n):
                if freq[nums[j]] == 0:
                    distinct_count += 1
                freq[nums[j]] += 1
                
                if distinct_count == total_distinct:
                    count += 1
        
        return count