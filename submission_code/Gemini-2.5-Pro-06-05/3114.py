class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_total_sum = 0
        
        # We can iterate through each possible index 'i' and consider it as the peak
        # of the mountain. For each choice of peak, we construct the tallest
        # possible beautiful configuration of towers and calculate its sum. The
        # maximum of these sums over all possible peaks will be the answer.
        
        for i in range(n):
            # Let the i-th tower be the peak. To maximize the sum, its height should be
            # as large as possible, so we set heights[i] = maxHeights[i].
            peak_height = maxHeights[i]
            current_sum = peak_height
            
            # --- Calculate the sum for the left side of the peak (indices 0 to i-1) ---
            # The heights must be non-decreasing up to the peak. To maximize the sum,
            # we make each height as large as possible while satisfying the constraints:
            # 1. heights[j] <= maxHeights[j]
            # 2. heights[j] <= heights[j+1]
            
            # We start from the tower next to the peak (i-1) and move leftwards.
            last_height = peak_height
            for j in range(i - 1, -1, -1):
                # The height at j is limited by maxHeights[j] and the height of the
                # tower to its right (which we have already determined).
                last_height = min(maxHeights[j], last_height)
                current_sum += last_height
            
            # --- Calculate the sum for the right side of the peak (indices i+1 to n-1) ---
            # The heights must be non-increasing from the peak.
            # Constraints:
            # 1. heights[j] <= maxHeights[j]
            # 2. heights[j] <= heights[j-1]
            
            # We start from the tower next to the peak (i+1) and move rightwards.
            last_height = peak_height
            for j in range(i + 1, n):
                # The height at j is limited by maxHeights[j] and the height of the
                # tower to its left.
                last_height = min(maxHeights[j], last_height)
                current_sum += last_height
            
            # Update the overall maximum sum found so far.
            max_total_sum = max(max_total_sum, current_sum)
            
        return max_total_sum