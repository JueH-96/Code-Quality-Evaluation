class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 1000000007
        n = len(nums)
        dp_prev = [1] * (nums[0] + 1)
        
        for idx in range(1, n):
            diff = max(0, nums[idx] - nums[idx - 1])
            # Compute cumulative sum for dp_prev
            len_prev = len(dp_prev)
            cum_sum = [0] * len_prev
            cum_sum[0] = dp_prev[0] % MOD
            for v in range(1, len_prev):
                cum_sum[v] = (cum_sum[v - 1] + dp_prev[v]) % MOD
            # Now dp_curr
            max_val_next = nums[idx]
            dp_curr = [0] * (max_val_next + 1)
            for val_next in range(max_val_next + 1):
                if val_next - diff >= 0:
                    ub = min(nums[idx - 1], val_next - diff)
                    dp_curr[val_next] = cum_sum[ub]
                else:
                    dp_curr[val_next] = 0
            # Update dp_prev
            dp_prev = dp_curr
        
        # Sum dp_prev with modulo
        total = 0
        for val in dp_prev:
            total = (total + val) % MOD
        return total