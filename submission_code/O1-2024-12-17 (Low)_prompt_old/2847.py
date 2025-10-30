class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        word_set = set(words)
        pairs = 0
        
        for w in words:
            if w in word_set:
                rev = w[::-1]
                # Look for a distinct reversed match
                if rev in word_set and w != rev:
                    pairs += 1
                    # Remove both to prevent reuse
                    word_set.remove(w)
                    word_set.remove(rev)
        
        return pairs