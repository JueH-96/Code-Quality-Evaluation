class Solution:
    def countCompleteSubarrays(self, nums: List[int]) -> int:
        distinct_count = len(set(nums))
        n = len(nums)
        count = 0
        
        for start in range(n):
            seen = set()
            for end in range(start, n):
                seen.add(nums[end])
                if len(seen) == distinct_count:
                    count += n - end
                    break
        
        return count