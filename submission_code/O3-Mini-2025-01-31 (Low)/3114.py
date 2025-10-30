from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        best = 0
        
        # Try each index i as the peak.
        for peak in range(n):
            # assign the peak its maximum possible height (since higher peak always helps sum)
            peak_height = maxHeights[peak]
            
            # Build left side: non-decreasing (going from left to peak)
            curr = peak_height
            left_sum = 0
            # iterate backwards from peak-1 to 0, each tower's height is min(maxHeights[j], height of tower to its right)
            for j in range(peak-1, -1, -1):
                curr = min(maxHeights[j], curr)
                left_sum += curr
            
            # Build right side: non-increasing (from peak to right)
            curr = peak_height
            right_sum = 0
            # iterate forwards from peak+1 to n-1, each tower's height is min(maxHeights[j], height of tower to its left)
            for j in range(peak+1, n):
                curr = min(maxHeights[j], curr)
                right_sum += curr
            
            total = left_sum + peak_height + right_sum
            best = max(best, total)
            
        return best

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.maximumSumOfHeights([5,3,4,1,1]))  # Expected output: 13
    # Example 2:
    print(sol.maximumSumOfHeights([6,5,3,9,2,7]))  # Expected output: 22
    # Example 3:
    print(sol.maximumSumOfHeights([3,2,5,5,2,3]))  # Expected output: 18