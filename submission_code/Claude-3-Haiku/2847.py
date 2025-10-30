class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        # Create a dictionary to store the reversed strings and their indices
        reversed_words = {}
        for i, word in enumerate(words):
            reversed_word = word[::-1]
            if reversed_word in reversed_words:
                reversed_words[reversed_word].append(i)
            else:
                reversed_words[reversed_word] = [i]
        
        # Count the number of pairs
        pairs = 0
        for word in words:
            reversed_word = word[::-1]
            if reversed_word in reversed_words:
                pairs += len(reversed_words[reversed_word])
                reversed_words[reversed_word] = []
        
        # Divide the total number of pairs by 2 to get the final result
        return pairs // 2