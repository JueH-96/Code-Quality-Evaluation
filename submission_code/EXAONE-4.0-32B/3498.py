class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        total_pairs = n // 2
        A = [0] * (k + 1)
        cnt_M = [0] * (k + 2)
        
        for i in range(total_pairs):
            x = nums[i]
            y = nums[n - 1 - i]
            base = abs(x - y)
            if base <= k:
                A[base] += 1
            
            M_val = max(max(x, k - x), max(y, k - y))
            if M_val <= k:
                cnt_M[M_val] += 1
        
        B_arr = [0] * (k + 2)
        for d in range(k, -1, -1):
            B_arr[d] = cnt_M[d] + B_arr[d + 1]
        
        min_changes = float('inf')
        for d in range(0, k + 1):
            cost = 2 * total_pairs - (A[d] + B_arr[d])
            if cost < min_changes:
                min_changes = cost
                
        return min_changes