class Solution:
    def minOperations(self, nums: List[int], k: int) -> int:
        # Track which numbers from 1 to k we still need
        needed = set(range(1, k + 1))
        
        # Process elements from right to left (simulating removal from end)
        for i in range(len(nums) - 1, -1, -1):
            # If this element is one we need, mark it as collected
            if nums[i] in needed:
                needed.remove(nums[i])
            
            # If we've collected all numbers from 1 to k, return operations count
            if not needed:
                return len(nums) - i
        
        # This line should never be reached given the problem constraints
        return len(nums)