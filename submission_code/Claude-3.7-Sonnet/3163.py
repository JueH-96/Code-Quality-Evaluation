class Solution:
    def sumCounts(self, nums: List[int]) -> int:
        n = len(nums)
        result = 0
        
        for i in range(n):
            # For each starting position i
            distinct_elements = set()
            for j in range(i, n):
                # For each ending position j
                distinct_elements.add(nums[j])
                # Add the square of the distinct count to our result
                result += len(distinct_elements) ** 2
        
        return result