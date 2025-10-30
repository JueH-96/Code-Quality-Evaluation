from typing import List

class Solution:
    def maximumTotalSum(self, maximumHeight: List[int]) -> int:
        n = len(maximumHeight)
        if len(set(maximumHeight)) < n:
            return -1
        
        maximumHeight.sort()
        prefix_sum = [0] * (n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + maximumHeight[i]
        
        max_sum = -1
        for i in range(n):
            target_height = (i + 1) * (maximumHeight[i] - i) // 2
            if target_height > maximumHeight[i]:
                continue
            left = 0
            right = i
            while left <= right:
                mid = (left + right) // 2
                if (mid + 1) * (2 * maximumHeight[i] - mid) // 2 <= prefix_sum[i + 1] - prefix_sum[mid]:
                    left = mid + 1
                else:
                    right = mid - 1
            max_sum = max(max_sum, prefix_sum[left] + (i + 1) * (maximumHeight[i] - left) // 2)
        
        return max_sum