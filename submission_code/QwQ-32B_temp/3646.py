class Solution:
    def sumOfGoodSubsequences(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        sum_dp = {}
        cnt_dp = {}
        
        for num in nums:
            temp_sum = num
            temp_cnt = 1
            
            # Check for num-1
            prev_num = num - 1
            if prev_num in sum_dp:
                s = sum_dp[prev_num]
                c = cnt_dp[prev_num]
                temp_sum += s + c * num
                temp_cnt += c
            
            # Check for num+1
            next_num = num + 1
            if next_num in sum_dp:
                s = sum_dp[next_num]
                c = cnt_dp[next_num]
                temp_sum += s + c * num
                temp_cnt += c
            
            # Update sum_dp and cnt_dp for current num
            prev_s = sum_dp.get(num, 0)
            prev_c = cnt_dp.get(num, 0)
            new_s = (prev_s + temp_sum) % MOD
            new_c = (prev_c + temp_cnt) % MOD
            sum_dp[num] = new_s
            cnt_dp[num] = new_c
        
        # Sum all values in sum_dp to get the total
        total = sum(sum_dp.values()) % MOD
        return total