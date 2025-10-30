class Solution:
    def medianOfUniquenessArray(self, nums: List[int]) -> int:
        n = len(nums)
        uniqueness_array = []
        
        # Calculate the number of distinct elements in each subarray
        for i in range(n):
            distinct_set = set()
            for j in range(i, n):
                distinct_set.add(nums[j])
                uniqueness_array.append(len(distinct_set))
        
        # Sort the uniqueness array
        uniqueness_array.sort()
        
        # Find the median
        m = len(uniqueness_array)
        if m % 2 == 1:
            return uniqueness_array[m // 2]
        else:
            return uniqueness_array[m // 2 - 1]