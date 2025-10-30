class Solution:
    def maxValue(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # Initialize first_dp: first_dp[a][i] is a bitmask for possible OR values
        first_dp = [[0] * (n + 1) for _ in range(k + 1)]
        first_dp[0][0] = 1  # 0 elements, OR is 0 (bit 0 is set)
        
        for i in range(1, n + 1):
            first_dp[0][i] = 1  # a=0, OR remains 0
        
        for a in range(1, k + 1):
            for i in range(1, n + 1):
                # Not taking the current element (i-1 in nums)
                first_dp[a][i] = first_dp[a][i-1]
                if a > 0:
                    prev_mask = first_dp[a-1][i-1]
                    num = nums[i-1]
                    new_mask = 0
                    for bit in range(128):
                        if (prev_mask >> bit) & 1:
                            new_or = bit | num
                            new_mask |= 1 << new_or
                    first_dp[a][i] |= new_mask
        
        # Initialize second_dp: second_dp[i][b] is the max OR for selecting b elements starting at i
        second_dp = [[0] * (k + 1) for _ in range(n + 1)]
        
        for i in range(n - 1, -1, -1):
            for b in range(1, k + 1):
                option1 = nums[i] | second_dp[i + 1][b - 1]
                option2 = second_dp[i + 1][b]
                second_dp[i][b] = max(option1, option2)
        
        max_val = 0
        # Iterate over possible i where first part ends and second can start
        for i in range(k, n - k + 1):
            mask = first_dp[k][i]
            if mask == 0:
                continue
            or2 = second_dp[i][k]
            for or1 in range(128):
                if (mask >> or1) & 1:
                    current_xor = or1 ^ or2
                    if current_xor > max_val:
                        max_val = current_xor
        
        return max_val