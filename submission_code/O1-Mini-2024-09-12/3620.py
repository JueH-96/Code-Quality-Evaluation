class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        sorted_nums = sorted(nums)
        count = 0
        last_assigned = float('-inf')
        
        for num in sorted_nums:
            lower = num - k
            upper = num + k
            assign = max(lower, last_assigned + 1)
            if assign <= upper:
                count += 1
                last_assigned = assign
        
        return count