class Solution:
    def countCompleteSubstrings(self, word: str, k: int) -> int:
        def check(length):
            count = [0] * 26
            left = 0
            for right in range(len(word)):
                count[ord(word[right]) - ord('a')] += 1
                if right - left + 1 > length:
                    count[ord(word[left]) - ord('a')] -= 1
                    left += 1
                if all(c == 0 or c == k for c in count):
                    nonlocal result
                    result += 1
        
        result = 0
        for length in range(1, len(word) // k + 1):
            check(length)
        return result