class Solution:
    def isGood(self, nums: List[int]) -> bool:
        if not nums:
            return False
        
        n = max(nums)
        
        # Check if length is correct for base[n]
        if len(nums) != n + 1:
            return False
        
        # Count occurrences of each number
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        
        # Check if we have the right counts
        # Numbers 1 to n-1 should appear exactly once
        for i in range(1, n):
            if count.get(i, 0) != 1:
                return False
        
        # Number n should appear exactly twice
        if count.get(n, 0) != 2:
            return False
        
        # Check that no other numbers are present
        # The total count should be exactly n+1 numbers from 1 to n
        expected_total = (n - 1) + 2  # (n-1) numbers appearing once + n appearing twice
        actual_total = sum(count.values())
        
        return actual_total == expected_total and len(count) == n