class Solution:
    def minBitwiseArray(self, nums: List[int]) -> List[int]:
        def find_smallest(num):
            for i in range(num):
                if i | (i + 1) == num:
                    return i
            return -1
        
        return [find_smallest(num) for num in nums]