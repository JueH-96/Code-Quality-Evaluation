class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10**9 + 7
        n = len(s)
        # Map creatures to indices: F=0, W=1, E=2
        mp = {'F': 0, 'W': 1, 'E': 2}
        a = [mp[ch] for ch in s]
        # Precompute win[b][a]: 1 if b beats a, -1 if a beats b, 0 otherwise
        win = [[0]*3 for _ in range(3)]
        beats = {(0,2),  # F beats E
                 (1,0),  # W beats F
                 (2,1)}  # E beats W
        for b in range(3):
            for aa in range(3):
                if (b, aa) in beats:
                    win[b][aa] = 1
                elif (aa, b) in beats:
                    win[b][aa] = -1
                else:
                    win[b][aa] = 0
        
        # dp_prev[b][d] = number of ways up to previous round,
        # last Bob move = b, score difference = d - offset
        offset = n
        size = 2*n + 1
        dp_prev = [[0]*size for _ in range(3)]
        
        # Initialize for round 0
        for b in range(3):
            d0 = win[b][a[0]] + offset
            dp_prev[b][d0] = 1
        
        # Iterate rounds 1..n-1
        for i in range(1, n):
            dp_curr = [[0]*size for _ in range(3)]
            ai = a[i]
            for prev_b in range(3):
                row = dp_prev[prev_b]
                # if no sequences with this last move, skip
                # but we'll check inside
                for d_idx in range(size):
                    cnt = row[d_idx]
                    if cnt == 0:
                        continue
                    cur_diff = d_idx - offset
                    # choose new b != prev_b
                    for b in range(3):
                        if b == prev_b:
                            continue
                        nd = cur_diff + win[b][ai]
                        nd_idx = nd + offset
                        # valid index always in [0..2n]
                        dp_curr[b][nd_idx] = (dp_curr[b][nd_idx] + cnt) % MOD
            dp_prev = dp_curr
        
        # Sum all sequences with final diff > 0
        ans = 0
        for b in range(3):
            for d_idx in range(offset+1, size):
                ans = (ans + dp_prev[b][d_idx]) % MOD
        
        return ans