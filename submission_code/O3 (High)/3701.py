from typing import List

class Solution:
    def minCostGoodCaption(self, caption: str) -> str:
        n = len(caption)
        # Every run must be at least 3, so length<3  ⇒ impossible
        if n < 3:
            return ""
        
        INF = 10**9          #  > maximum possible cost (n*25 <= 1_250_000)
        ALPHA = 26
        size = (n + 1) * ALPHA
        
        # indices helper
        def idx(pos: int, ch: int) -> int:
            return pos * ALPHA + ch
        
        # DP arrays:
        # dp1[pos][c] – minimal cost to finish caption[pos:]
        #               when current run before pos is (char=c , length=1)
        # dp2[pos][c] – ... length=2
        # dp3[pos][c] – ... length>=3   (stored as 3)
        # They are stored flattened inside three big lists.
        dp1 = [INF] * size
        dp2 = [INF] * size
        dp3 = [INF] * size
        
        # Base  :  past the end  (pos == n)
        # run length <3  ⇒ impossible
        base = n * ALPHA
        for c in range(ALPHA):
            dp3[base + c] = 0          # a completed run needs no more cost
            # dp1 , dp2 already INF
        
        # work from right to left
        for pos in range(n - 1, -1, -1):
            base_cur  = pos * ALPHA
            base_next = (pos + 1) * ALPHA
            letter_ord = ord(caption[pos])
            
            # Pre–compute distance from every letter to caption[pos]
            dist = [abs(letter_ord - (97 + c)) for c in range(ALPHA)]
            
            # values needed when we switch to a NEW run (len becomes 1)
            best1_val = INF
            best2_val = INF
            best1_char = -1
            for c in range(ALPHA):
                val = dist[c] + dp1[base_next + c]     # cost if we put c here and its run length will be 1
                if val < best1_val:
                    best2_val = best1_val
                    best1_val = val
                    best1_char = c
                elif val < best2_val:
                    best2_val = val
            
            # fill dp for current position
            for c in range(ALPHA):
                # continue a run of length 1  -> becomes length 2
                dp1[base_cur + c] = dist[c] + dp2[base_next + c]
                
                # run length 2 -> becomes 3
                dp2[base_cur + c] = dist[c] + dp3[base_next + c]
                
                # currently at length>=3
                keep_same = dist[c] + dp3[base_next + c]          # continue same run
                # or switch to some other character (run length 1)
                switch_cost = best1_val if c != best1_char else best2_val
                dp3[base_cur + c] = keep_same if keep_same < switch_cost else switch_cost
        
        # Minimal total cost starting from "no previous character" (treated as run length 3 so we may pick any letter)
        letter0_ord = ord(caption[0])
        min_cost = INF
        for c in range(ALPHA):
            d = abs(letter0_ord - (97 + c))
            cand = d + dp1[ALPHA + c]        # after char at pos0, run length will be 1, thus look at dp1 for pos=1
            if cand < min_cost:
                min_cost = cand
        
        if min_cost >= INF:
            return ""           # impossible
        
        # Greedy reconstruction of lexicographically smallest string with cost == min_cost
        res: List[str] = []
        prefix_cost = 0
        prev_char = ALPHA        # sentinel for "no previous char"
        run_len   = 3            # sentinel treated as a completed run
        
        for pos in range(n):
            letter_ord = ord(caption[pos])
            chosen = -1
            chosen_cost = 0
            chosen_new_len = 0
            
            for c in range(ALPHA):
                # check if we are allowed to place c here
                if prev_char != ALPHA and c != prev_char and run_len < 3:
                    continue      # cannot switch yet
                d = abs(letter_ord - (97 + c))
                
                # determine new run length
                new_len = run_len + 1 if c == prev_char else 1
                if new_len > 3:
                    new_len = 3
                
                # cost of the suffix
                if pos == n - 1:
                    # last position; suffix cost is 0 only if run complete
                    suffix = 0 if new_len == 3 else INF
                else:
                    base_next = (pos + 1) * ALPHA + c
                    if new_len == 1:
                        suffix = dp1[base_next]
                    elif new_len == 2:
                        suffix = dp2[base_next]
                    else:
                        suffix = dp3[base_next]
                
                if suffix >= INF:
                    continue
                
                total = prefix_cost + d + suffix
                if total == min_cost:
                    chosen = c
                    chosen_cost = d
                    chosen_new_len = new_len
                    break    # first (smallest) letter that keeps minimal cost
        
            # Safety (should never fail)
            if chosen == -1:
                return ""
            
            res.append(chr(97 + chosen))
            prefix_cost += chosen_cost
            prev_char = chosen
            run_len = chosen_new_len
        
        return ''.join(res)