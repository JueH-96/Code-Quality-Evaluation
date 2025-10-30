class Solution:
    def minimumDeletions(self, word: str, k: int) -> int:
        freq = [0]*26
        for c in word:
            freq[ord(c) - ord('a')] += 1
        deletions = 0
        for c in word:
            if freq[ord(c) - ord('a')] < k:
                deletions += 1
            freq[ord(c) - ord('a')] -= 1
        return deletions