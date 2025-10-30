from typing import List
import bisect

class Solution:
    def maxPathLength(self, coordinates: List[List[int]], k: int) -> int:
        xk, yk = coordinates[k]
        left = []
        right = []
        
        for i, (x, y) in enumerate(coordinates):
            if i == k:
                continue
            if x < xk and y < yk:
                left.append((x, y))
            elif x > xk and y > yk:
                right.append((x, y))
        
        # Process left group
        left.sort()
        left_y = [y for x, y in left]
        left_lis = self.compute_LIS(left_y)
        
        # Process right group
        right.sort()
        right_y = [y for x, y in right]
        right_lis = self.compute_LIS(right_y)
        
        return left_lis + right_lis + 1
    
    def compute_LIS(self, nums: List[int]) -> int:
        tails = []
        for num in nums:
            idx = bisect.bisect_left(tails, num)
            if idx == len(tails):
                tails.append(num)
            else:
                tails[idx] = num
        return len(tails)