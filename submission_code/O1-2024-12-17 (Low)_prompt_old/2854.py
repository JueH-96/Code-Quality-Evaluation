class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        """
        We want to perform n-1 "join" operations on an array of n words. At step i,
        we choose either join(str_{i-1}, words[i]) or join(words[i], str_{i-1]),
        where join(x,y) concatenates x and y, but if x's last character equals y's
        first character, one of them is deleted in the concatenation.

        Our goal is to minimize the length of the final string str_{n-1}.
        """

        n = len(words)
        # If there's only one word, answer is simply len(words[0])
        if n == 1:
            return len(words[0])

        # Precompute for each word: first char index, last char index, length
        # We'll represent 'a'..'z' as 0..25
        def char_to_int(c: str) -> int:
            return ord(c) - ord('a')
        
        first = [char_to_int(w[0]) for w in words]
        last  = [char_to_int(w[-1]) for w in words]
        lengths = [len(w) for w in words]

        # DP[i][fc][lc] = minimal length after processing up to i-th word (0-based),
        # resulting in a string whose first character index is fc, last character index is lc.
        # We'll use a large number to represent "unreachable".
        INF = 10**9
        DP = [[[INF]*26 for _ in range(26)] for _ in range(n)]
        
        # Initialize DP for the 0-th word
        DP[0][first[0]][last[0]] = lengths[0]
        
        # Fill DP for subsequent words
        for i in range(1, n):
            f = first[i]
            l = last[i]
            length_i = lengths[i]
            for fc in range(26):
                for lc in range(26):
                    old_len = DP[i-1][fc][lc]
                    if old_len == INF:
                        continue

                    # Option 1: join(str_{i-1}, words[i])
                    #   new first char = fc
                    #   new last char = l
                    #   new length = old_len + length_i - (1 if lc == f else 0)
                    cost1 = old_len + length_i
                    if lc == f:
                        cost1 -= 1
                    new_fc1 = fc
                    new_lc1 = l
                    DP[i][new_fc1][new_lc1] = min(DP[i][new_fc1][new_lc1], cost1)

                    # Option 2: join(words[i], str_{i-1})
                    #   new first char = f
                    #   new last char = lc
                    #   new length = old_len + length_i - (1 if l == fc else 0)
                    cost2 = old_len + length_i
                    if l == fc:
                        cost2 -= 1
                    new_fc2 = f
                    new_lc2 = lc
                    DP[i][new_fc2][new_lc2] = min(DP[i][new_fc2][new_lc2], cost2)
        
        # The answer is the minimum of DP[n-1][fc][lc] over all possible fc, lc
        ans = INF
        for fc in range(26):
            for lc in range(26):
                ans = min(ans, DP[n-1][fc][lc])
        
        return ans