class Solution:
    def minimumCost(self, nums: List[int], cost: List[int], k: int) -> int:
        n = len(nums)
        
        # Precompute prefix sums for efficiency
        nums_prefix = [0] * (n + 1)
        cost_prefix = [0] * (n + 1)
        
        for i in range(n):
            nums_prefix[i + 1] = nums_prefix[i] + nums[i]
            cost_prefix[i + 1] = cost_prefix[i] + cost[i]
        
        # dp[i] = minimum cost to divide nums[0..i-1]
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            # Try all possible starting positions for the last subarray
            for j in range(i):
                # Last subarray is nums[j..i-1]
                # This would be subarray number (number of subarrays ending before j + 1)
                subarrays_before = 0
                temp_pos = j
                while temp_pos > 0:
                    # Count how many subarrays we had before position j
                    # We can determine this by checking our DP transitions
                    subarrays_before += 1
                    temp_pos -= 1  # This is a simplified counting, we need better approach
                
                # Actually, let's track the subarray number differently
                # The subarray number is determined by how we got to position j
                # Let's modify our approach
                
        # Better approach: track both position and number of subarrays
        # dp[i][subarrays] = min cost to reach position i with exactly 'subarrays' subarrays
        
        # Actually, let's use a simpler DP where we track the subarray count implicitly
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        
        for i in range(1, n + 1):
            for j in range(i):
                # Calculate number of subarrays if we make nums[j..i-1] the last subarray
                # We need to find optimal way to divide nums[0..j-1] first
                if dp[j] == float('inf'):
                    continue
                
                # Count subarrays in optimal division of nums[0..j-1]
                # This is tricky - let's use a different approach
                
        # Let me restart with a cleaner approach
        # dp[i] stores (min_cost, num_subarrays) for nums[0..i-1]
        dp = [(0, 0)] + [(float('inf'), 0)] * n
        
        for i in range(1, n + 1):
            for j in range(i):
                if dp[j][0] == float('inf'):
                    continue
                
                prev_cost, prev_subarrays = dp[j]
                current_subarray_num = prev_subarrays + 1
                
                # Cost of subarray nums[j..i-1] as the current_subarray_num-th subarray
                nums_sum = nums_prefix[i]  # sum of nums[0..i-1]
                cost_sum = cost_prefix[i] - cost_prefix[j]  # sum of cost[j..i-1]
                
                subarray_cost = (nums_sum + k * current_subarray_num) * cost_sum
                total_cost = prev_cost + subarray_cost
                
                if total_cost < dp[i][0]:
                    dp[i] = (total_cost, current_subarray_num)
        
        return dp[n][0]