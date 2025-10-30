class Solution:
    def numberOfSpecialChars(self, word: str) -> int:
        # Create two sets for lowercase and uppercase letters found in the word.
        lower_set = set()
        upper_set = set()
        
        # Iterate over each character in the word.
        for ch in word:
            if ch.islower():
                lower_set.add(ch)
            elif ch.isupper():
                upper_set.add(ch)
                
        # Convert the uppercase set to lowercase for comparison.
        # The special letter is one that has both lowercase and its corresponding uppercase present.
        special_count = 0
        for ch in lower_set:
            if ch.upper() in upper_set:
                special_count += 1
                
        return special_count

# Example tests
if __name__ == "__main__":
    solution = Solution()
    
    # Test cases
    print(solution.numberOfSpecialChars("aaAbcBC"))  # Expected output: 3
    print(solution.numberOfSpecialChars("abc"))      # Expected output: 0
    print(solution.numberOfSpecialChars("abBCab"))   # Expected output: 1