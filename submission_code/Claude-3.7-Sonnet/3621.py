class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # If any element is less than k, it's impossible to make all elements equal to k
        if any(num < k for num in nums):
            return -1
        
        # Count the number of distinct values greater than k
        return len(set(num for num in nums if num > k))