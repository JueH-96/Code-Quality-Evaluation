class Solution:
    def longestString(self, x: int, y: int, z: int) -> int:
        # dp[a][b][c][lc][rl] = reachable state using a "AA", b "BB", c "AB",
        # ending with last_char=lc and run_len=rl
        # last_char: 0='A',1='B',2=None; run_len: 0,1,2
        dp = [[[[[False]*3 for _ in range(3)]
                 for _ in range(3)]
                for _ in range(z+1)]
               for _ in range(y+1)]
        
        # start with no blocks, no last char, run_len=0
        dp[0][0][0][2][0] = True
        
        for b in range(y+1):
            for c in range(z+1):
                for a in range(x+1):
                    for lc in range(3):
                        for rl in range(3):
                            if not dp[b][c][a][lc][rl]:
                                continue
                            # try add "AA"
                            if a < x:
                                # can't form "AAA"
                                # after adding 2 A's, run_len -> 2
                                # need run_len+2 <= 2 if lc=='A'
                                if not (lc == 0 and rl >= 1):
                                    dp[b][c][a+1][0][2] = True
                            # try add "BB"
                            if b < y:
                                # can't form "BBB"
                                if not (lc == 1 and rl >= 1):
                                    dp[b+1][c][a][1][2] = True
                            # try add "AB"
                            if c < z:
                                # for "AB", first char A then B
                                # check we don't get 3 A's in a row:
                                # first yields run1 = rl+1 if lc=='A' else 1
                                # must be <=2 => forbid lc=='A' and rl==2
                                if not (lc == 0 and rl == 2):
                                    # after AB, last char is B with run_len=1
                                    dp[b][c+1][a][1][1] = True
        
        # find max blocks used
        max_blocks = 0
        for b in range(y+1):
            for c in range(z+1):
                for a in range(x+1):
                    for lc in range(3):
                        for rl in range(3):
                            if dp[b][c][a][lc][rl]:
                                max_blocks = max(max_blocks, a + b + c)
        # each block is length 2
        return max_blocks * 2