class Solution:
    def isGood(self, nums: List[int]) -> bool:
        # Find the maximum element which should be n
        n = max(nums)
        
        # Check if the length is correct (should be n + 1)
        if len(nums) != n + 1:
            return False
        
        # Count occurrences of each number
        count = {}
        for num in nums:
            count[num] = count.get(num, 0) + 1
        
        # Check that we have exactly one occurrence of 1 to n-1
        for i in range(1, n):
            if count.get(i, 0) != 1:
                return False
        
        # Check that we have exactly two occurrences of n
        if count.get(n, 0) != 2:
            return False
        
        # Check that there are no other numbers
        for num in count:
            if num < 1 or num > n:
                return False
        
        return True