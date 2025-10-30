class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Check if there are any elements less than k
        if any(num < k for num in nums):
            return -1
        
        # Find all unique values greater than k
        unique_greater = set(num for num in nums if num > k)
        
        # The number of operations is the count of unique elements greater than k
        return len(unique_greater)