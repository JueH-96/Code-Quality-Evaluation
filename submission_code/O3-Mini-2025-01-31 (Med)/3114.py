from typing import List

class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        best = 0
        
        # For each possible peak index "peak", compute the best mountain configuration
        # that uses that index as the peak.
        for peak in range(n):
            # Build the left segment (from index 0 to peak) with non-decreasing heights
            left = [0] * (peak + 1)
            # For the peak, we choose the highest possible value which is maxHeights[peak]
            left[peak] = maxHeights[peak]
            # Move leftwards ensuring that at each step the height is the highest possible
            # but not more than the tower's max and not more than its right neighbor to keep 
            # the union non-decreasing (when read from left to peak).
            for i in range(peak - 1, -1, -1):
                left[i] = min(maxHeights[i], left[i + 1])
            left_sum = sum(left)
            
            # Build the right segment (from index peak to n-1) with non-increasing heights
            right = [0] * (n - peak)
            right[0] = maxHeights[peak]
            # Move rightwards ensuring that each tower's height does not exceed its left neighbor,
            # preserving the mountain proper (non-increasing on the right side).
            for j in range(1, n - peak):
                idx = peak + j
                right[j] = min(maxHeights[idx], right[j - 1])
            right_sum = sum(right)
            
            # The peak is counted in both segments; subtract its value once.
            total = left_sum + right_sum - maxHeights[peak]
            best = max(best, total)
        
        return best

# Sample testing code:
if __name__ == '__main__':
    sol = Solution()
    print(sol.maximumSumOfHeights([5,3,4,1,1]))   # Expected output: 13
    print(sol.maximumSumOfHeights([6,5,3,9,2,7]))   # Expected output: 22
    print(sol.maximumSumOfHeights([3,2,5,5,2,3]))   # Expected output: 18