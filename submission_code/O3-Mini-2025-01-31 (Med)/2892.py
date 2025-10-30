class Solution:
    def isGood(self, nums: List[int]) -> bool:
        # The candidate n should be the maximum element in the array.
        n = max(nums)
        # The good array based on n should have exactly n+1 elements.
        if len(nums) != n + 1:
            return False
        
        # Count frequency of each number in nums.
        freq = {}
        for num in nums:
            freq[num] = freq.get(num, 0) + 1
        
        # For the array to be "good", it must contain:
        # - Exactly one occurrence of each number from 1 to n-1.
        # - Exactly two occurrences of n.
        for i in range(1, n):
            if freq.get(i, 0) != 1:
                return False
        
        if freq.get(n, 0) != 2:
            return False
        
        # Ensure that there are no extra numbers other than 1 through n.
        for key in freq:
            if key < 1 or key > n:
                return False
        
        return True