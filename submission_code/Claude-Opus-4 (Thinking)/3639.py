class Solution:
    def isZeroArray(self, nums: List[int], queries: List[List[int]]) -> bool:
        n = len(nums)
        # Count how many times each position can be decremented
        count = [0] * (n + 1)  # Extra position for difference array
        
        # Mark the start and end of each query range
        for l, r in queries:
            count[l] += 1
            count[r + 1] -= 1
        
        # Calculate prefix sum to get actual counts at each position
        for i in range(1, n):
            count[i] += count[i - 1]
        
        # Check if each position can be decremented enough times
        for i in range(n):
            if nums[i] > count[i]:
                return False
        
        return True