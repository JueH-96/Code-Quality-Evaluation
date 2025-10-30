class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        count = 0
        for i in range(n):
            freq = [0]*26
            for j in range(i, n):
                freq[ord(word[j])-ord('a')] += 1
                if freq[ord(word[j])-ord('a')] > k:
                    break
                if freq[ord(word[j])-ord('a')] == k:
                    if all(val == 0 or val == k for val in freq):
                        min_char = max_char = word[j]
                        for l in range(26):
                            if freq[l] == k:
                                min_char = min(min_char, chr(ord('a')+l))
                                max_char = max(max_char, chr(ord('a')+l))
                        if ord(max_char) - ord(min_char) <= 2:
                            count += 1
        return count