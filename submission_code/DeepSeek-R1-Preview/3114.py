class Solution:
    def maximumSumOfHeights(self, maxHeights: List[int]) -> int:
        n = len(maxHeights)
        max_sum = 0
        for i in range(n):
            # Compute left array from 0 to i
            left = [0] * (i + 1)
            left[i] = maxHeights[i]
            for j in range(i - 1, -1, -1):
                left[j] = min(maxHeights[j], left[j + 1])
            # Compute right array from i to n-1
            right = [0] * (n - i)
            right[0] = maxHeights[i]
            for j in range(i + 1, n):
                idx = j - i
                right[idx] = min(maxHeights[j], right[idx - 1])
            # Calculate sum
            sum_left = sum(left)
            sum_right = sum(right[1:])  # exclude the peak which is already in sum_left
            total = sum_left + sum_right
            if total > max_sum:
                max_sum = total
        return max_sum