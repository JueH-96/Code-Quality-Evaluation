class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # If any element is less than k, it's impossible
        if any(num < k for num in nums):
            return -1
        
        # Get all distinct values greater than k
        distinct_values = set()
        for num in nums:
            if num > k:
                distinct_values.add(num)
        
        # The number of operations is the number of distinct values greater than k
        return len(distinct_values)