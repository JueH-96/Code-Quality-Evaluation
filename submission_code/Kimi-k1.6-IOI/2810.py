class Solution:
    def minCost(self, nums: List[int], x: int) -> int:
        n = len(nums)
        min_arrays = []
        
        for j in range(n):
            min_arr = [0] * n
            min_arr[0] = nums[j]
            for k in range(1, n):
                prev = (j - k) % n
                min_arr[k] = min(min_arr[k-1], nums[prev])
            min_arrays.append(min_arr)
        
        min_total = float('inf')
        for k in range(n):
            current_sum = sum(min_arrays[j][k] for j in range(n))
            current_total = current_sum + k * x
            if current_total < min_total:
                min_total = current_total
        
        return min_total