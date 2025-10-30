class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        if not nums:
            return 0
        max_val = max(nums)
        max_size = max_val + 2  # To cover up to x+1 which is max_val + 1
        cnt = [0] * max_size
        sum_val = [0] * max_size
        
        for x in nums:
            x_prev = x - 1
            x_next = x + 1
            
            # Get previous counts and sums, handling x_prev < 0
            prev_cnt_prev = cnt[x_prev] if x_prev >= 0 else 0
            prev_sum_prev = sum_val[x_prev] if x_prev >= 0 else 0
            
            prev_cnt_next = cnt[x_next]
            prev_sum_next = sum_val[x_next]
            
            # Calculate new contributions from this x
            new_count = 1 + prev_cnt_prev + prev_cnt_next
            new_sum = x + (prev_sum_prev + x * prev_cnt_prev) + (prev_sum_next + x * prev_cnt_next)
            
            # Update the current x's count and sum with modulo
            cnt[x] = (cnt[x] + new_count) % MOD
            sum_val[x] = (sum_val[x] + new_sum) % MOD
        
        # Sum all sum_val entries and return modulo MOD
        total = sum(sum_val) % MOD
        return total