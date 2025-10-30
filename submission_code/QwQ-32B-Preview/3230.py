class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        n = len(word)
        if n == 0:
            return 0
        
        # Function to check if two characters are almost equal
        def almost_equal(char1, char2):
            return char1 == char2 or abs(ord(char1) - ord(char2)) == 1
        
        # Initialize previous DP state
        prev_dp = [0] * 26
        for j in range(26):
            char_j = chr(j + ord('a'))
            prev_dp[j] = 0 if word[0] == char_j else 1
        
        # Iterate through the string
        for i in range(1, n):
            current_dp = [0] * 26
            for j in range(26):
                char_j = chr(j + ord('a'))
                cost = 0 if word[i] == char_j else 1
                min_prev = float('inf')
                for k in range(26):
                    char_k = chr(k + ord('a'))
                    if not almost_equal(char_j, char_k):
                        min_prev = min(min_prev, prev_dp[k])
                current_dp[j] = cost + min_prev
            prev_dp = current_dp
        
        return min(prev_dp)