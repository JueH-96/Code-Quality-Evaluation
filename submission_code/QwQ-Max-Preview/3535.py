class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        
        # Initialize DP for the first element
        prev_dp = [0] * (nums[0] + 1)
        for a in range(nums[0] + 1):
            prev_dp[a] = 1
        
        for i in range(1, n):
            current_num = nums[i]
            prev_num = nums[i-1]
            delta = max(0, current_num - prev_num)
            
            # Create a difference array for the current step
            diff = [0] * (current_num + 2)
            
            for a_prev in range(prev_num + 1):
                if a_prev >= len(prev_dp) or prev_dp[a_prev] == 0:
                    continue
                min_current = a_prev + delta
                if min_current > current_num:
                    continue
                contribution = prev_dp[a_prev]
                diff[min_current] = (diff[min_current] + contribution) % MOD
                if current_num + 1 <= current_num + 1:
                    diff[current_num + 1] = (diff[current_num + 1] - contribution) % MOD
            
            # Compute the current DP array using the difference array
            current_dp = [0] * (current_num + 1)
            current_sum = 0
            for a in range(current_num + 1):
                current_sum = (current_sum + diff[a]) % MOD
                current_dp[a] = current_sum
            
            prev_dp = current_dp
        
        return sum(prev_dp) % MOD