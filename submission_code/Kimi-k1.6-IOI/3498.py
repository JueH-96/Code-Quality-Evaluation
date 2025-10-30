class Solution:
    def minChanges(self, nums: List[int], k: int) -> int:
        n = len(nums)
        m = n // 2
        diff_count = {}
        max_vals = []
        
        for i in range(m):
            a = nums[i]
            b = nums[n - 1 - i]
            d = abs(a - b)
            diff_count[d] = diff_count.get(d, 0) + 1
            max_val = max(k - b, b, k - a, a)
            max_vals.append(max_val)
        
        # Create max_freq array
        max_freq = [0] * (k + 2)  # indices 0 to k+1
        for val in max_vals:
            if val <= k:
                max_freq[val] += 1
        
        # Compute right_sum
        right_sum = [0] * (k + 2)
        right_sum[k] = max_freq[k]
        for x in range(k - 1, -1, -1):
            right_sum[x] = max_freq[x] + right_sum[x + 1]
        
        max_total = 0
        for X in range(k + 1):
            count_d = diff_count.get(X, 0)
            count_le_max_val = right_sum[X]
            current_total = count_d + count_le_max_val
            if current_total > max_total:
                max_total = current_total
        
        return 2 * m - max_total