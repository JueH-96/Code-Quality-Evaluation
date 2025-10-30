class Solution:
    def minimizeConcatenatedLength(self, words: List[str]) -> int:
        # Initialize the current string as the first word
        current_str = words[0]
        
        # Iterate over the words from the second word to the last
        for i in range(1, len(words)):
            word = words[i]
            
            # Calculate the length if we join current_str with word
            if current_str[-1] == word[0]:
                len1 = len(current_str) + len(word) - 1
            else:
                len1 = len(current_str) + len(word)
            
            # Calculate the length if we join word with current_str
            if word[-1] == current_str[0]:
                len2 = len(word) + len(current_str) - 1
            else:
                len2 = len(word) + len(current_str)
            
            # Choose the minimum length option
            if len1 <= len2:
                if current_str[-1] == word[0]:
                    current_str += word[1:]
                else:
                    current_str += word
            else:
                if word[-1] == current_str[0]:
                    current_str = word + current_str[1:]
                else:
                    current_str = word + current_str
        
        # Return the length of the final string
        return len(current_str)