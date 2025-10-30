class Solution:
    def isPossibleToSplit(self, nums: List[int]) -> bool:
        n = len(nums)
        counts = {}
        for num in nums:
            counts[num] = counts.get(num, 0) + 1
        
        for count in counts.values():
            if count > n // 2:
                return False
        
        return True