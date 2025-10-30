class Solution:
    def maximumLength(self, nums: List[int]) -> int:
        even = []
        odd = []
        for num in nums:
            if num % 2 == 0:
                even.append(num)
            else:
                odd.append(num)
        
        if len(even) == 0 or len(odd) == 0:
            return len(nums)
        
        return max(len(even), len(odd))