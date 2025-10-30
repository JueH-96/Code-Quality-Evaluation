class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        prefix_mins = []
        for t in range(n):
            arr = []
            for s in range(n):
                pos = (t - s) % n
                arr.append(nums[pos])
            current_min = float('inf')
            pmin = []
            for num in arr:
                if num < current_min:
                    current_min = num
                pmin.append(current_min)
            prefix_mins.append(pmin)
        
        min_total = float('inf')
        for k in range(n):
            current_sum = 0
            for t in range(n):
                current_sum += prefix_mins[t][k]
            total = current_sum + k * x
            if total < min_total:
                min_total = total
        return min_total