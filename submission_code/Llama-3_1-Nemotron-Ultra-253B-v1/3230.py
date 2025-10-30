class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        if n == 0:
            return 0
        word_ords = [ord(c) - ord('a') for c in word]
        prev_dp = [float('inf')] * 26
        first_char = word_ords[0]
        for c in range(26):
            prev_dp[c] = 0 if c == first_char else 1
        for i in range(1, n):
            curr_dp = [float('inf')] * 26
            current_char = word_ords[i]
            for c_curr in range(26):
                cost = 0 if c_curr == current_char else 1
                min_prev = float('inf')
                for c_prev in range(26):
                    if (c_prev != c_curr) and (abs(c_prev - c_curr) > 1):
                        if prev_dp[c_prev] < min_prev:
                            min_prev = prev_dp[c_prev]
                if min_prev != float('inf'):
                    curr_dp[c_curr] = min_prev + cost
            prev_dp = curr_dp
        return min(prev_dp)