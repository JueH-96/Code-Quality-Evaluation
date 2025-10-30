class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        seen = set(words)
        ans = 0
        
        for word in words:
            rev = word[::-1]
            if rev in seen and word != rev:
                # Found a valid pair, remove both from seen to avoid reuse.
                ans += 1
                seen.remove(word)
                seen.remove(rev)
        return ans