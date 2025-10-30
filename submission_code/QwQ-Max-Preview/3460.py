MOD = 10**9 + 7

class Solution:
    def numberOfPermutations(self, n: int, requirements: List[List[int]]) -> int:
        # Sort requirements by end_i
        req_sorted = sorted(requirements, key=lambda x: x[0])
        # Create a dictionary to quickly check required counts
        required = {}
        for end, cnt in req_sorted:
            required[end] = cnt
        
        # Precompute binomial coefficients up to n
        max_n = max(n, 400 + n)  # Adjust based on possible maximum
        comb = [[0] * (max_n + 1) for _ in range(max_n + 1)]
        comb[0][0] = 1
        for i in range(1, max_n + 1):
            comb[i][0] = 1
            for j in range(1, i + 1):
                comb[i][j] = (comb[i-1][j-1] + comb[i-1][j]) % MOD
        
        # Initialize DP
        dp = [dict() for _ in range(n)]
        first_end = req_sorted[0][0] if req_sorted else -1
        if 0 in required:
            if required[0] != 0:
                return 0
            dp[0][0] = 1
        else:
            dp[0][0] = 1
        
        for k in range(n-1):
            current_end = k
            if current_end in required:
                current_cnt = required[current_end]
                if current_cnt not in dp[k]:
                    return 0
                current_ways = dp[k][current_cnt]
            else:
                current_ways = sum(dp[k].values()) % MOD
            
            next_end = k + 1
            next_required = required.get(next_end, None)
            
            for c in dp[k]:
                if current_end in required and c != required[current_end]:
                    continue
                ways = dp[k][c]
                for t in range(0, k + 2):
                    new_c = c + t
                    if new_c > 400:  # Assuming cnt_i is up to 400
                        continue
                    if next_required is not None and new_c != next_required:
                        continue
                    if new_c not in dp[k+1]:
                        dp[k+1][new_c] = 0
                    # Compute the number of x's that contribute t inversions
                    # Using the combinatorial identity C(n, k+1)
                    cnt = comb[n][k+1]
                    dp[k+1][new_c] = (dp[k+1][new_c] + ways * cnt) % MOD
            # If next_end is a required end, filter the DP
            if next_end in required:
                allowed_c = required[next_end]
                total = 0
                if allowed_c in dp[k+1]:
                    total = dp[k+1][allowed_c]
                dp[k+1] = {allowed_c: total}
        
        last_end = n - 1
        if last_end not in required:
            return sum(dp[n-1].values()) % MOD
        else:
            final_c = required[last_end]
            return dp[n-1].get(final_c, 0) % MOD