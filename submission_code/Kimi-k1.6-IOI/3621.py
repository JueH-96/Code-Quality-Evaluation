class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Check if any element is less than k
        for num in nums:
            if num < k:
                return -1
        
        # Collect elements greater than k
        greater = [num for num in nums if num > k]
        
        # The number of unique elements greater than k is the answer
        return len(set(greater))