class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        n = len(nums)
        uniqueness_array = []
        
        for i in range(n):
            seen = set()
            for j in range(i, n):
                seen.add(nums[j])
                uniqueness_array.append(len(seen))
        
        uniqueness_array.sort()
        
        mid = len(uniqueness_array) // 2
        return uniqueness_array[mid]