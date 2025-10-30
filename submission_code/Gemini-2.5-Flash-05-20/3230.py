class Solution:
    def removeAlmostEqualCharacters(self, word: str) -> int:
        operations = 0
        n = len(word)
        
        # Iterate through the word using index i.
        # We need to check characters at i and i+1, so i should go up to n-2.
        # The while loop condition `i < n - 1` ensures `i+1` is always a valid index.
        i = 0
        while i < n - 1:
            # Get the ASCII values of the current character and the next character.
            char1_ord = ord(word[i])
            char2_ord = ord(word[i+1])
            
            # Check if the characters are almost-equal.
            # Two characters 'a' and 'b' are almost-equal if abs(ord(a) - ord(b)) <= 1.
            if abs(char1_ord - char2_ord) <= 1:
                # If they are almost-equal, we must perform an operation.
                # Increment the total operations count.
                operations += 1
                
                # When we perform an operation on word[i+1], we can change it to any lowercase
                # English letter. This means we can pick a character that not only resolves
                # the conflict with word[i] but also avoids a conflict with word[i+2] (if it exists).
                # For example, if word[i] is 'a' and word[i+2] is 'b', we could change word[i+1]
                # to 'd'. 'a' and 'd' are not almost-equal, and 'd' and 'b' are not almost-equal.
                # Since we can always find such a character (26 letters, at most 6 forbidden),
                # this single operation effectively resolves conflicts for two adjacent pairs:
                # (word[i], word[i+1]) and (word[i+1], word[i+2]).
                # Therefore, we can skip the next character check by advancing 'i' by 2.
                i += 2
            else:
                # If the characters are not almost-equal, no operation is needed for this pair.
                # Move to the next adjacent pair by advancing 'i' by 1.
                i += 1
                
        return operations