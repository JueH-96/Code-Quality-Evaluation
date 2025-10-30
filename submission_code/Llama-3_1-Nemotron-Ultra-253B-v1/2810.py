from typing import List

class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Precompute reversed cyclic lists for each type t
        reversed_lists = []
        for t in range(n):
            rl = []
            for m in range(n):
                pos = (t - m) % n
                rl.append(nums[pos])
            reversed_lists.append(rl)
        
        # Precompute minima for each t and k
        minima = [[0] * n for _ in range(n)]
        for t in range(n):
            current_min = float('inf')
            for k in range(n):
                current_min = min(current_min, reversed_lists[t][k])
                minima[t][k] = current_min
        
        # Calculate the minimum total cost for each possible k_ops
        min_total = float('inf')
        for k_ops in range(n):
            total = 0
            for t in range(n):
                total += minima[t][k_ops]
            total += k_ops * x
            if total < min_total:
                min_total = total
        
        return min_total