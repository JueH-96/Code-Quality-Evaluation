class Solution:
    def maximumNumberOfStringPairs(self, words: List[str]) -> int:
        # Create a set to store the words that can form pairs
        word_set = set(words)
        # Initialize the count of pairs
        pair_count = 0
        
        # Iterate over each word in the list
        for word in words:
            # Reverse the current word
            reversed_word = word[::-1]
            # Check if the reversed word is in the set
            if reversed_word in word_set:
                # Increment the pair count
                pair_count += 1
                # Remove both the word and its reverse from the set
                word_set.remove(word)
                word_set.remove(reversed_word)
        
        return pair_count