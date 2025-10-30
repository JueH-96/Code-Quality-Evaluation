class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        seen = set()
        pairs = 0
        
        for word in words:
            reversed_word = word[::-1]
            if reversed_word in seen:
                pairs += 1
            else:
                seen.add(word)
        
        return pairs