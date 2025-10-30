class Solution:
    def maxPalindromesAfterOperations(self, words: List[str]) -> int:
        def is_palindrome(s):
            return s == s[::-1]

        n = len(words)
        dp = [0] * (1 << n)
        for mask in range(1, 1 << n):
            bits = [i for i in range(n) if (mask >> i) & 1]
            if len(bits) == 1:
                word = words[bits[0]]
                dp[mask] = int(is_palindrome(word))
            else:
                i, j = bits[0], bits[1]
                word1, word2 = words[i], words[j]
                for x in range(len(word1)):
                    for y in range(len(word2)):
                        new_word1 = word1[:x] + word2[y] + word1[x + 1:]
                        new_word2 = word2[:y] + word1[x] + word2[y + 1:]
                        if is_palindrome(new_word1) and is_palindrome(new_word2):
                            dp[mask] = max(dp[mask], 1 + dp[mask ^ (1 << i) ^ (1 << j)])
        return max(dp)