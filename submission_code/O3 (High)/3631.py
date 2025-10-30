MOD = 10 ** 9 + 7

class Solution:
    def countKReducibleNumbers(self, s: str, k: int) -> int:
        L = len(s)                                       # amount of bits of n
        
        # -------------------------------------------------------------
        # 1.  depth[t] = how many times we must apply popcount to reach 1
        # -------------------------------------------------------------
        max_c = L                                        # popcount(x) never exceeds L
        depth = [-1] * (max_c + 1)
        depth[1] = 0                                     # base
        
        def pc(x: int) -> int:
            return bin(x).count('1')
        
        # memoised recursion
        def get_depth(v: int) -> int:
            if depth[v] != -1:
                return depth[v]
            depth[v] = 1 + get_depth(pc(v))
            return depth[v]
        
        for v in range(2, max_c + 1):
            get_depth(v)
        
        # -------------------------------------------------------------
        # 2.  which pop-count values make a number k-reducible ?
        #     for x>1 : depth(x) = 1 + depth(popcount(x))
        #     we need 1 + depth(popcount(x)) <= k   →   depth(popcount(x)) <= k-1
        # -------------------------------------------------------------
        allowed = set(c for c in range(1, max_c + 1) if depth[c] <= k - 1)
        
        # -------------------------------------------------------------
        # 3.  digit DP over the bits of n to count numbers < n
        #     with popcount in `allowed`
        # -------------------------------------------------------------
        # dp_equal[c]   : number of prefixes that are still equal to n so far
        # dp_less [c]   : number of prefixes that are already <  n
        dp_equal = [0] * (L + 1)
        dp_less  = [0] * (L + 1)
        dp_equal[0] = 1
        
        for pos, ch in enumerate(s):                     # from most to least significant
            bit = int(ch)
            nxt_equal = [0] * (L + 1)
            nxt_less  = [0] * (L + 1)
            
            for ones in range(pos + 1):                  # at most `pos` ones have been set so far
                if dp_less[ones]:
                    val = dp_less[ones]
                    # choose 0 on this bit
                    nxt_less[ones]   = (nxt_less[ones]   + val) % MOD
                    # choose 1 on this bit
                    nxt_less[ones+1] = (nxt_less[ones+1] + val) % MOD
                
                if dp_equal[ones]:
                    val = dp_equal[ones]
                    if bit == 0:
                        # only 0 keeps equality
                        nxt_equal[ones] = (nxt_equal[ones] + val) % MOD
                    else:  # bit == 1
                        # choose 0  → number becomes smaller
                        nxt_less[ones] = (nxt_less[ones] + val) % MOD
                        # choose 1  → still equal
                        nxt_equal[ones+1] = (nxt_equal[ones+1] + val) % MOD
            
            dp_equal, dp_less = nxt_equal, nxt_less
        
        # -------------------------------------------------------------
        # 4.  collect results: only states that are already < n (dp_less)
        # -------------------------------------------------------------
        ans = 0
        for c in allowed:
            ans = (ans + dp_less[c]) % MOD
        return ans