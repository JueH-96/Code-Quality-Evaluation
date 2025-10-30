class Solution:
    def maxDistinctElements(self, nums: List[int], k: int) -> int:
        nums.sort()
        last = -float('inf')
        count = 0
        for x in nums:
            lower = x - k
            upper = x + k
            possible_min = max(last + 1, lower)
            if possible_min > upper:
                assigned = upper
            else:
                assigned = possible_min
                last = assigned
                count += 1
        return count