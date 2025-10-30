class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        diff = [0] * (n + 1)
        
        # Process all queries
        for l, r in queries:
            diff[l] += 1
            diff[r + 1] -= 1
        
        # Calculate the cumulative sum of decrements
        decrements = 0
        for i in range(n):
            decrements += diff[i]
            if decrements > nums[i]:
                return False
            nums[i] -= decrements
        
        # Check if all elements are zero
        return all(num == 0 for num in nums)