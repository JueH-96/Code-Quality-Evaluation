class Solution:
    def validSequence(self, word1: str, word2: str) -> List[int]:
        n = len(word2)
        m = len(word1)
        if n > m:
            return []
        
        # Initialize the result list
        result = []
        # Initialize the mismatch count
        mismatch = 0
        # Initialize the pointer for word1
        ptr1 = 0
        
        for i in range(n):
            # Find the next character in word1 that matches word2[i]
            while ptr1 < m and word1[ptr1] != word2[i]:
                ptr1 += 1
            if ptr1 >= m:
                # No more characters to match
                return []
            # Check if we need to change the character
            if word1[ptr1] != word2[i]:
                mismatch += 1
                if mismatch > 1:
                    return []
            # Add the index to the result
            result.append(ptr1)
            # Move to the next character in word1
            ptr1 += 1
        
        # Check if the sequence is valid
        # We need to ensure that the sequence is the lex smallest
        # So we need to find the earliest possible indices
        # The above logic already ensures that
        
        return result