class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        n = len(words)
        INF = 10**9
        dp = [[INF] * 26 for _ in range(26)]
        
        first_word = words[0]
        a0 = ord(first_word[0]) - ord('a')
        b0 = ord(first_word[-1]) - ord('a')
        dp[a0][b0] = len(first_word)
        
        for i in range(1, n):
            w = words[i]
            w_start = ord(w[0]) - ord('a')
            w_end = ord(w[-1]) - ord('a')
            len_w = len(w)
            
            new_dp = [[INF] * 26 for _ in range(26)]
            
            for a in range(26):
                for b in range(26):
                    if dp[a][b] == INF:
                        continue
                    current_len = dp[a][b]
                    
                    # Option 1: Append w to current string
                    overlap1 = 1 if b == w_start else 0
                    total_len1 = current_len + len_w - overlap1
                    if total_len1 < new_dp[a][w_end]:
                        new_dp[a][w_end] = total_len1
                    
                    # Option 2: Prepend w to current string
                    overlap2 = 1 if w_end == a else 0
                    total_len2 = len_w + current_len - overlap2
                    if total_len2 < new_dp[w_start][b]:
                        new_dp[w_start][b] = total_len2
            
            dp = new_dp
        
        ans = INF
        for a in range(26):
            for b in range(26):
                if dp[a][b] < ans:
                    ans = dp[a][b]
        return ans