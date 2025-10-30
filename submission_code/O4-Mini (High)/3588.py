class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        # Map characters to integers
        mp = {'F': 0, 'W': 1, 'E': 2}
        s_nums = [mp[ch] for ch in s]
        # Precompute outcome delta: dtbl[b][a] = +1 if b beats a,
        # -1 if b loses to a, 0 if tie.
        dtbl = [[0]*3 for _ in range(3)]
        for b in range(3):
            for a in range(3):
                d = (b - a) % 3
                if d == 1:
                    dtbl[b][a] = 1
                elif d == 2:
                    dtbl[b][a] = -1
        # DP arrays: dp0, dp1, dp2 for last move = 0,1,2
        # We shift diff by 'offset' so diffs in [-n..n] map to [0..2n]
        offset = n
        size = 2*offset + 1
        dp0 = [0] * size
        dp1 = [0] * size
        dp2 = [0] * size
        # Initialize for the first round
        a0 = s_nums[0]
        d0 = dtbl[0][a0]; dp0[offset + d0] = 1
        d1 = dtbl[1][a0]; dp1[offset + d1] = 1
        d2 = dtbl[2][a0]; dp2[offset + d2] = 1
        # Current diff-range after round 1 is [-1..1] -> indices [offset-1..offset+1]
        low, high = offset - 1, offset + 1
        
        # Iterate through rounds 2..n
        for i in range(1, n):
            ai = s_nums[i]
            # Precompute the three possible deltas for this round
            dd0 = dtbl[0][ai]
            dd1 = dtbl[1][ai]
            dd2 = dtbl[2][ai]
            # New DP arrays
            dp0n = [0] * size
            dp1n = [0] * size
            dp2n = [0] * size
            new_low, new_high = low - 1, high + 1
            
            # Transitions from last=0 -> next in {1,2}
            arr = dp0
            for idx in range(low, high + 1):
                v = arr[idx]
                if v:
                    j1 = idx + dd1
                    tmp = dp1n[j1] + v
                    if tmp >= MOD: tmp -= MOD
                    dp1n[j1] = tmp
                    j2 = idx + dd2
                    tmp = dp2n[j2] + v
                    if tmp >= MOD: tmp -= MOD
                    dp2n[j2] = tmp
            
            # Transitions from last=1 -> next in {0,2}
            arr = dp1
            for idx in range(low, high + 1):
                v = arr[idx]
                if v:
                    j0 = idx + dd0
                    tmp = dp0n[j0] + v
                    if tmp >= MOD: tmp -= MOD
                    dp0n[j0] = tmp
                    j2 = idx + dd2
                    tmp = dp2n[j2] + v
                    if tmp >= MOD: tmp -= MOD
                    dp2n[j2] = tmp
            
            # Transitions from last=2 -> next in {0,1}
            arr = dp2
            for idx in range(low, high + 1):
                v = arr[idx]
                if v:
                    j0 = idx + dd0
                    tmp = dp0n[j0] + v
                    if tmp >= MOD: tmp -= MOD
                    dp0n[j0] = tmp
                    j1 = idx + dd1
                    tmp = dp1n[j1] + v
                    if tmp >= MOD: tmp -= MOD
                    dp1n[j1] = tmp
            
            # Move to next round
            dp0, dp1, dp2 = dp0n, dp1n, dp2n
            low, high = new_low, new_high
        
        # Sum over all sequences where final diff > 0
        # diff index > offset up to offset + n
        ans = 0
        start = offset + 1
        end = offset + n
        for idx in range(start, end + 1):
            ans = (ans + dp0[idx] + dp1[idx] + dp2[idx]) % MOD
        return ans