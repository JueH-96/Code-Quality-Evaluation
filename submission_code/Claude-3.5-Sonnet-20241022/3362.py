class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        uniqueness = []
        
        # Generate all possible subarrays and count distinct elements
        for i in range(n):
            seen = set()
            for j in range(i, n):
                seen.add(nums[j])
                uniqueness.append(len(seen))
        
        # Sort the uniqueness array
        uniqueness.sort()
        
        # Find median
        m = len(uniqueness)
        if m % 2 == 0:
            # If even length, take smaller of two middle values
            return uniqueness[m//2 - 1]
        else:
            return uniqueness[m//2]