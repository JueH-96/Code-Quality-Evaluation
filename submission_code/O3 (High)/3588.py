class Solution:
    def countWinningSequences(self, s: str) -> int:
        MOD = 10 ** 9 + 7
        n = len(s)
        
        # Mapping of characters to indices
        char_id = {'F': 0, 'W': 1, 'E': 2}
        
        # Pre-compute the score difference Bob gets for every (alice, bob) pair
        # diff = +1 : Bob wins the round
        # diff =  0 : draw
        # diff = -1 : Alice wins the round
        diff_tbl = [[0] * 3 for _ in range(3)]
        for alice in range(3):
            for bob in range(3):
                if bob == alice:
                    diff = 0
                elif (bob - alice + 3) % 3 == 1:   # Bob beats Alice
                    diff = 1
                else:                               # Alice beats Bob
                    diff = -1
                diff_tbl[alice][bob] = diff
        
        # dp[last][d] – number of sequences up to current position
        #               that end with Bob's creature 'last'
        #               and have total score difference = d - shift  (range [-n … n])
        shift = n                      # offset so we can store negative differences
        size  = 2 * n + 1              # indices 0 … 2n  represent differences −n … +n
        dp = [[0] * size for _ in range(3)]
        
        # Initialization – first round
        alice0 = char_id[s[0]]
        for bob in range(3):
            diff = diff_tbl[alice0][bob]
            dp[bob][shift + diff] = 1
        
        # Process the remaining rounds
        for i in range(1, n):
            alice = char_id[s[i]]
            nxt = [[0] * size for _ in range(3)]
            
            for last in range(3):                       # Bob's creature in the previous round
                row = dp[last]
                for idx, cnt in enumerate(row):         # enumerate all stored differences
                    if cnt == 0:
                        continue
                    for bob in range(3):                # Bob's choice for this round
                        if bob == last:                 # cannot repeat the creature
                            continue
                        ndiff = idx + diff_tbl[alice][bob]
                        nxt[bob][ndiff] = (nxt[bob][ndiff] + cnt) % MOD
            dp = nxt
        
        # Sum sequences whose final score difference is positive
        answer = 0
        for last in range(3):
            for idx in range(shift + 1, size):          # indices representing diff > 0
                answer = (answer + dp[last][idx]) % MOD
        
        return answer