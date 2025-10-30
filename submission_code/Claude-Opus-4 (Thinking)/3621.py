class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Check if any element is less than k
        if min(nums) < k:
            return -1
        
        # Count unique values greater than k
        return len(set(num for num in nums if num > k))