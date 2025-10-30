from collections import defaultdict

class Solution:
    def longestValidSubstring(self, word: str, forbidden: List[str]) -> int:
        if not word or not forbidden: 
            return len(word)
        
        last_index = defaultdict(list)
        
        # Build a dictionary to hold all indexes for each forbidden substring
        for f in forbidden:
            for i in range(len(f)):
                last_index[f[i]].append((f, i))

        max_len = 0
        dp = [0] * len(word)
        
        # Calculate the longest valid substring length for each character in the word
        for i in range(len(word) - 1, -1, -1):
            c = word[i]
            if not last_index.get(c): 
                dp[i] = 1
            else:
                dp[i] = 1
                len_i = 1

                # Check for all possible forbidden substrings ending with the current character
                for f, j in last_index.get(c):
                    if j == 0 and i + len(f) <= len(word) and word[i:i+len(f)] == f: 
                        len_i = 0
                        break

                    elif j != 0 and i + len_i + (len(f) - j - 1) <= len(word) and word[i-len_i:i+len_i+len(f)-j-1] == f[:j] + f[j+1:]:
                        len_i += dp[i + len_i] + len(f) - j - 1
                        break

                dp[i] = len_i

            max_len = max(max_len, dp[i])

        return max_len