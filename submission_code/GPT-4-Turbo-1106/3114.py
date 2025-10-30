class Solution:
    def maximumSumOfHeights(self, maxHeights: list[int]) -> int:
        n = len(maxHeights)
        # Create two prefix arrays to store the minimum height from left and right
        left = [0] * n
        right = [0] * n
        
        # Fill the left array
        left[0] = maxHeights[0]
        for i in range(1, n):
            left[i] = min(left[i-1] + 1, maxHeights[i])
        
        # Fill the right array
        right[-1] = maxHeights[-1]
        for i in range(n-2, -1, -1):
            right[i] = min(right[i+1] + 1, maxHeights[i])
        
        # Calculate the maximum sum by taking the minimum of left and right arrays
        max_sum = 0
        for i in range(n):
            max_sum += min(left[i], right[i])
        
        return max_sum

# Example usage:
# sol = Solution()
# print(sol.maximumSumOfHeights([5,3,4,1,1])) # Output: 13
# print(sol.maximumSumOfHeights([6,5,3,9,2,7])) # Output: 22
# print(sol.maximumSumOfHeights([3,2,5,5,2,3])) # Output: 18