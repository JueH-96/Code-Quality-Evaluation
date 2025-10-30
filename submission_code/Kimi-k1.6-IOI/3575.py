class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        if n == 0 or k == 0:
            return 0
        
        # Precompute left_dps: left_dps[i] represents the state after processing the first i elements
        left_dps = []
        current_dp = [0] * (k + 1)
        current_dp[0] = 1 << 0  # Initial state: 0 elements selected, OR is 0
        left_dps.append(current_dp.copy())
        
        for num in nums:
            new_dp = [0] * (k + 1)
            for s in range(k + 1):
                new_dp[s] = current_dp[s]
                if s >= 1:
                    mask = 0
                    for x in range(128):
                        if current_dp[s - 1] & (1 << x):
                            val = x | num
                            mask |= (1 << val)
                    new_dp[s] |= mask
            current_dp = new_dp
            left_dps.append(current_dp.copy())
        
        # Precompute right_dps: right_dps[i] represents the state for elements from i to n-1
        right_dps = [[0] * (k + 1) for _ in range(n + 1)]
        current_dp = [0] * (k + 1)
        current_dp[0] = 1 << 0  # Initial state: no elements selected, OR is 0
        right_dps[n] = current_dp.copy()
        
        for i in range(n - 1, -1, -1):
            num = nums[i]
            new_dp = [0] * (k + 1)
            for s in range(k + 1):
                new_dp[s] = current_dp[s]
                if s >= 1:
                    mask = 0
                    for x in range(128):
                        if current_dp[s - 1] & (1 << x):
                            val = x | num
                            mask |= (1 << val)
                    new_dp[s] |= mask
            current_dp = new_dp
            right_dps[i] = current_dp.copy()
        
        max_result = 0
        # Iterate over all valid split points
        for i in range(k - 1, n - k):
            left_mask = left_dps[i + 1][k]
            right_mask = right_dps[i + 1][k]
            
            a_list = [x for x in range(128) if (left_mask & (1 << x))]
            b_list = [y for y in range(128) if (right_mask & (1 << y))]
            
            if not a_list or not b_list:
                continue
            
            current_max = 0
            for a in a_list:
                for b in b_list:
                    current_max = max(current_max, a ^ b)
            
            if current_max > max_result:
                max_result = current_max
        
        return max_result