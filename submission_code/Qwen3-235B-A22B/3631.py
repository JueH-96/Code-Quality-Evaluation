class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        mod = 10**9 + 7
        bits = list(map(int, s))
        m = len(bits)
        
        # Precompute steps for each c from 1 to m
        step_list = [0] * (m + 2)  # covers up to m+1
        for c in range(1, m + 2):
            steps = 0
            curr = c
            while curr != 1:
                curr = bin(curr).count('1')
                steps += 1
            step_list[c] = steps
        
        # Initialize DP table
        dp = [[[0] * 2 for _ in range(m + 2)] for __ in range(m + 2)]
        dp[0][0][1] = 1  # Starting state
        
        for pos in range(m):
            for cnt in range(m + 1):
                for tight in [0, 1]:
                    current = dp[pos][cnt][tight]
                    if not current:
                        continue
                    
                    # Determine allowed bits based on tight constraint
                    allowed_b = []
                    if tight:
                        max_bit = bits[pos]
                        for b in [0, 1]:
                            if b <= max_bit:
                                allowed_b.append(b)
                    else:
                        allowed_b = [0, 1]
                    
                    # Process each allowed bit
                    for b in allowed_b:
                        new_tight_flag = tight and (b == bits[pos])
                        new_tight = 1 if new_tight_flag else 0
                        new_cnt = cnt + (b == 1)
                        
                        if new_cnt > m:
                            continue
                        
                        dp[pos + 1][new_cnt][new_tight] = (dp[pos + 1][new_cnt][new_tight] + current) % mod
        
        # Precompute count_le_s for all c
        count_le = [0] * (m + 2)
        count_s_bits = sum(bits)
        for c in range(m + 2):
            count_le[c] = (dp[m][c][0] + dp[m][c][1]) % mod
        
        ans = 0
        for c in range(1, m + 1):
            if step_list[c] <= k - 1:
                cl = count_le[c]
                if c == count_s_bits:
                    cl -= 1
                if cl < 0:
                    cl += mod
                ans = (ans + cl) % mod
        
        return ans