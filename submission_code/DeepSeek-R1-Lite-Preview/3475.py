class Solution:
    def minOperations(self, nums: List[int]) -> int:
        n = len(nums)
        if n < 3:
            # Cannot perform any flip operations
            if all(num == 1 for num in nums):
                return 0
            else:
                return -1
        
        flip = [0] * n
        # Initialize flip[0]
        flip[0] = (1 - nums[0]) % 2
        # Initialize flip[1]
        flip[1] = (1 - nums[1] - flip[0]) % 2
        
        # Compute flip[k] for k from 2 to n-3
        for k in range(2, n-2):
            flip[k] = (1 - nums[k] - flip[k-1] - flip[k-2]) % 2
        
        # For k from n-2 to n-1, set flip[k] = 0 and check the equation
        for k in range(n-2, n):
            # flip[k] = 0 (cannot perform flip starting at k)
            expected = (1 - nums[k]) % 2
            actual = (flip[k-2] + flip[k-1]) % 2
            if actual != expected:
                return -1
        
        # Sum up the flip operations that can be performed (from 0 to n-3)
        total_operations = sum(flip[:n-2])
        return total_operations