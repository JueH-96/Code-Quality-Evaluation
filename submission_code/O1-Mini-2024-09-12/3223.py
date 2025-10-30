class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        result = 0
        n = len(word)
        
        for left in range(n):
            freq = {}
            valid = True
            for right in range(left, n):
                if right > left and abs(ord(word[right]) - ord(word[right - 1])) > 2:
                    break
                c = word[right]
                freq[c] = freq.get(c, 0) + 1
                if freq[c] > k:
                    break
                # Check if all characters have exactly k counts
                if all(count == k for count in freq.values()):
                    result += 1
        return result