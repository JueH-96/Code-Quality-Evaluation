class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        MOD = 10**9 + 7
        n = len(nums)
        if n == 0:
            return 0
        
        # Initialize DP_prev for the first element
        dp_prev = [[0] * 51 for _ in range(51)]
        first_num = nums[0]
        for a in range(0, first_num + 1):
            b = first_num - a
            dp_prev[a][b] = 1
        
        for i in range(1, n):
            current_num = nums[i]
            # Compute suffix_b: suffix_b[a_prev][b] = sum_{k >= b} dp_prev[a_prev][k]
            suffix_b = [[0] * 51 for _ in range(51)]
            for a_prev in range(51):
                current_suffix = 0
                for b in range(50, -1, -1):
                    current_suffix += dp_prev[a_prev][b]
                    suffix_b[a_prev][b] = current_suffix
            
            # Compute prefix_a: prefix_a[b][a] = sum_{a_prev=0 to a} suffix_b[a_prev][b]
            prefix_a = [[0] * 51 for _ in range(51)]
            for b in range(51):
                current_sum = 0
                for a in range(51):
                    current_sum += suffix_b[a][b]
                    prefix_a[b][a] = current_sum
            
            # Compute current DP
            dp_current = [[0] * 51 for _ in range(51)]
            for a in range(0, current_num + 1):
                b_val = current_num - a
                dp_current[a][b_val] = prefix_a[b_val][a] % MOD
            
            dp_prev = dp_current
        
        # Sum all entries in the final DP
        total = 0
        for a in range(51):
            for b in range(51):
                total = (total + dp_prev[a][b]) % MOD
        return total