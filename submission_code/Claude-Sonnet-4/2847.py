class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        seen = set()
        pairs = 0
        
        for word in words:
            reversed_word = word[::-1]
            
            # If the reverse of current word is already seen, we found a pair
            if reversed_word in seen:
                pairs += 1
                # Remove the reversed word from seen since it's now paired
                seen.remove(reversed_word)
            else:
                # Add current word to seen for future matching
                seen.add(word)
        
        return pairs