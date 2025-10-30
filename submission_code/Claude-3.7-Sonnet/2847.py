class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        count = 0
        seen = set()
        
        for word in words:
            if word[::-1] in seen:
                count += 1
            seen.add(word)
        
        return count