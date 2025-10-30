class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        MOD = 10**9 + 7
        max_cnt = 400  # Maximum required inversion count as per problem constraints
        
        # Sort requirements by their end indices
        sorted_reqs = sorted(requirements, key=lambda x: x[0])
        req_dict = {end: cnt for end, cnt in sorted_reqs}
        
        # Initialize DP array for the first element (length 1)
        prev_dp = [0] * (max_cnt + 1)
        prev_dp[0] = 1  # Only 0 inversions possible
        
        for m in range(2, n + 1):
            current_end = m - 1
            current_dp = [0] * (max_cnt + 1)
            
            # Compute prefix sums of the previous DP array
            prefix = [0] * (max_cnt + 2)
            for i in range(max_cnt + 1):
                prefix[i + 1] = (prefix[i] + prev_dp[i]) % MOD
            
            # Calculate current DP values using the prefix sums
            for new_k in range(max_cnt + 1):
                a = new_k - (m - 1)
                b = new_k
                a = max(a, 0)
                b = min(b, max_cnt)
                
                if a > b:
                    current_dp[new_k] = 0
                else:
                    current_dp[new_k] = (prefix[b + 1] - prefix[a]) % MOD
            
            # Apply the requirement if current_end is in the requirements
            if current_end in req_dict:
                required = req_dict[current_end]
                # Zero out all counts except the required one
                for k in range(max_cnt + 1):
                    if k != required:
                        current_dp[k] = 0
            
            prev_dp = current_dp[:]  # Update previous DP to current DP
        
        # The final answer is the count for the last requirement (end = n-1)
        required_final = req_dict[n - 1]
        return prev_dp[required_final] % MOD