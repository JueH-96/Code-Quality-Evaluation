class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        if n == 0:
            return 0
        
        # Precompute the minimal cost for each type t considering up to m shifts
        min_for_t = [[0] * n for _ in range(n)]
        for t in range(n):
            current_min = float('inf')
            for m in range(n):
                idx = (t - m) % n
                current_val = nums[idx]
                if current_val < current_min:
                    current_min = current_val
                min_for_t[t][m] = current_min
        
        # Calculate the minimal total cost for each possible number of shifts (k)
        min_total = float('inf')
        for k in range(n):
            total = 0
            for t in range(n):
                total += min_for_t[t][k]
            total += x * k
            if total < min_total:
                min_total = total
        
        return min_total