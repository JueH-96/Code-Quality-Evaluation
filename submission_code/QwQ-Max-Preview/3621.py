class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Check if there's any element less than k, which makes it impossible
        if any(num < k for num in nums):
            return -1
        
        # Count the number of distinct elements greater than k
        distinct_elements_gt_k = len({num for num in nums if num > k})
        
        return distinct_elements_gt_k