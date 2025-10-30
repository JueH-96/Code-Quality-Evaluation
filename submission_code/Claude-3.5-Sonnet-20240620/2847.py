class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        pairs = set()
        count = 0
        
        for i, word in enumerate(words):
            reversed_word = word[::-1]
            if reversed_word in pairs:
                count += 1
            else:
                pairs.add(word)
        
        return count