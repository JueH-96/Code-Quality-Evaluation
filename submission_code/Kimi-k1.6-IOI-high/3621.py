class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Check if any element is less than k
        for num in nums:
            if num < k:
                return -1
        
        # Collect all unique elements greater than k
        unique_greater = set()
        for num in nums:
            if num > k:
                unique_greater.add(num)
        
        return len(unique_greater)