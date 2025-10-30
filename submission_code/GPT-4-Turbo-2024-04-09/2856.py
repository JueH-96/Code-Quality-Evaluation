class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        def count_distinct_elements(nums):
            return len(set(nums))
        
        total_distinct = count_distinct_elements(nums)
        n = len(nums)
        count = 0
        
        for start in range(n):
            seen = defaultdict(int)
            distinct_count = 0
            
            for end in range(start, n):
                if seen[nums[end]] == 0:
                    distinct_count += 1
                seen[nums[end]] += 1
                
                if distinct_count == total_distinct:
                    count += n - end
                    break
        
        return count