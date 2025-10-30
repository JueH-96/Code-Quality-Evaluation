class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        from collections import defaultdict
        
        n = len(nums)
        uniqueness_array = []
        
        # Iterate over all possible subarrays
        for i in range(n):
            seen = set()
            for j in range(i, n):
                seen.add(nums[j])
                uniqueness_array.append(len(seen))
        
        # Sort the uniqueness array
        uniqueness_array.sort()
        
        # Find the median
        length = len(uniqueness_array)
        if length % 2 == 1:
            # If odd, return the middle element
            median = uniqueness_array[length // 2]
        else:
            # If even, return the smaller of the two middle elements
            median = uniqueness_array[length // 2 - 1]
        
        return median