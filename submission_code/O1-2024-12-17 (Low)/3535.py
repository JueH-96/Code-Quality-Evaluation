class Solution:
    def countOfPairs(self, nums: List[int]) -> int:
        import sys
        input_data = sys.stdin.readline
        MOD = 10**9 + 7
        
        n = len(nums)
        max_val = 50  # given constraint nums[i] <= 50
        
        # dp[x][y] will represent the number of ways for the current i
        # such that arr1[i] = x, arr2[i] = y, and x+y = nums[i].
        # We'll alternate between dp_prev and dp_cur to save memory.
        
        # Initialize for i = 0
        dp_prev = [[0]*(max_val+1) for _ in range(max_val+1)]
        first_sum = nums[0]
        for x in range(first_sum+1):
            y = first_sum - x
            dp_prev[x][y] = 1
        
        def build_prefix_sum(arr):
            # Build prefix sums so that pref[x+1][y+1] = sum of arr[u][v] for u<=x, v<=y
            pref = [[0]*(max_val+2) for _ in range(max_val+2)]
            for i in range(max_val+1):
                row_sum = 0
                for j in range(max_val+1):
                    row_sum = (row_sum + arr[i][j]) % MOD
                    pref[i+1][j+1] = (pref[i][j+1] + row_sum) % MOD
            return pref
        
        def get_sum(pref, x, y):
            # Returns sum of dp[u][v] for 0 <= u <= x, 0 <= v <= y
            # using the prefix array (1-based indexing).
            if x < 0 or y < 0:
                return 0
            return pref[x+1][y+1]
        
        # Process i from 1..n-1
        for i in range(1, n):
            dp_cur = [[0]*(max_val+1) for _ in range(max_val+1)]
            # Build prefix for dp_prev
            pref = build_prefix_sum(dp_prev)
            
            s = nums[i]
            # For each x+y = s
            for x in range(s+1):
                y = s - x
                # sum_{x' <= x, y' >= y} dp_prev[x'][y']
                # which is sum_{x'=0..x}[ sum_{y'=y..max_val} dp_prev[x'][y'] ]
                # Using prefix sums: sum_{x'=0..x, y'=0..max_val} dp - sum_{x'=0..x, y'=0..(y-1)} dp
                total_up_to_x_50 = get_sum(pref, x, max_val)
                total_up_to_x_y_minus_1 = get_sum(pref, x, y-1)
                ways = (total_up_to_x_50 - total_up_to_x_y_minus_1) % MOD
                dp_cur[x][y] = ways
            
            dp_prev = dp_cur
        
        # The result is sum of dp_prev[x][y] for x+y = nums[n-1]
        ans = 0
        last_sum = nums[-1]
        for x in range(last_sum+1):
            y = last_sum - x
            ans = (ans + dp_prev[x][y]) % MOD
        
        return ans