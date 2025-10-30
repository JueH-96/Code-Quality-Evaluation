class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Check if impossible: any element less than k
        if any(num < k for num in nums):
            return -1
        
        # Count unique values greater than k
        unique_gt_k = set(num for num in nums if num > k)
        
        return len(unique_gt_k)