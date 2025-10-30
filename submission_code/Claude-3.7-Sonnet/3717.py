class Solution:
    def minOperations(self, nums: List[int], x: int, k: int) -> int:
        n = len(nums)
        # Compute cost to make each possible subarray of size x have equal elements
        costs = []
        for s in range(n - x + 1):
            subarray = sorted(nums[s:s+x])
            median = subarray[x // 2]
            cost = sum(abs(num - median) for num in subarray)
            costs.append(cost)
        
        # DP approach: dp[i][j] represents minimum cost to have i non-overlapping 
        # subarrays using elements up to index j
        prev_dp = [0] * n  # Base case: 0 cost for 0 subarrays
        
        for i in range(1, k + 1):
            curr_dp = [float('inf')] * n
            for j in range(x - 1, n):
                # Skip if not enough elements for i subarrays
                if j < i*x - 1:
                    continue
                
                # Option 1: Don't include index j in current subarray
                if j > x - 1:
                    curr_dp[j] = curr_dp[j-1]
                
                # Option 2: Include j as end of current subarray
                start_index = j - x + 1
                
                if i == 1:
                    curr_dp[j] = min(curr_dp[j], costs[start_index])
                else:
                    curr_dp[j] = min(curr_dp[j], prev_dp[j-x] + costs[start_index])
            
            prev_dp = curr_dp
        
        return prev_dp[n-1]