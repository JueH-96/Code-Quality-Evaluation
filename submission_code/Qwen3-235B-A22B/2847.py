class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        word_set = set(words)
        used = set()
        count = 0
        
        for word in words:
            if word in used:
                continue
            reversed_word = word[::-1]
            if reversed_word == word:
                continue
            if reversed_word in word_set and reversed_word not in used:
                count += 1
                used.add(word)
                used.add(reversed_word)
        
        return count