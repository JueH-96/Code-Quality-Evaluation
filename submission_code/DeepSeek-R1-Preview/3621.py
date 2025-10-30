class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Check if any element is less than k, making it impossible
        for num in nums:
            if num < k:
                return -1
        
        # Collect all unique elements greater than k
        unique_above_k = {num for num in nums if num > k}
        
        # The number of unique elements is the number of operations needed
        return len(unique_above_k)