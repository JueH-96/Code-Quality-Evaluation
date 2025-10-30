from typing import List

class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        word_count = {}
        pairs = 0
        
        # Count the occurrences of each word
        for word in words:
            if word in word_count:
                word_count[word] += 1
            else:
                word_count[word] = 1
        
        # Count the pairs
        for word in word_count:
            reversed_word = word[::-1]
            if reversed_word in word_count and word != reversed_word:
                pairs += min(word_count[word], word_count[reversed_word])
        
        # Since each pair is counted twice, divide the total count by 2
        pairs //= 2
        
        return pairs