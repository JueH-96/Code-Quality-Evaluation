from typing import List

class Solution:
    def isAcronym(self, words: List[str], s: str) -> bool:
        # The problem constraints guarantee:
        # - words is not empty (1 <= words.length <= 100).
        # - Each string in words is not empty (1 <= words[i].length <= 10).
        # - s is not empty (1 <= s.length <= 100).
        # - All characters are lowercase English letters.
        #
        # These constraints mean that accessing word[0] for any word in words
        # is always safe and will yield a character.

        # Method 1: Construct the acronym from 'words' and compare with 's'.
        # This is often concise and Pythonic.

        # Step 1: Construct the candidate acronym.
        # We can use a list comprehension to gather the first character of each word,
        # and then "".join() to form the string.
        
        # Example: words = ["alice", "bob", "charlie"]
        # first_chars will be ['a', 'b', 'c']
        first_chars = [word[0] for word in words]
        
        # candidate_acronym will be "abc"
        candidate_acronym = "".join(first_chars)
        
        # Step 2: Compare the constructed acronym with the given string 's'.
        # The string equality operator '==' checks if both strings are identical
        # in length and content.
        return candidate_acronym == s

        # Method 2: Iterative comparison (alternative).
        # This method avoids creating an intermediate list and string, potentially
        # being more memory-efficient for very large inputs, though less critical
        # given the current constraints (lengths up to 100).

        # # First, check if the number of words matches the length of s.
        # # An acronym must have exactly one character derived from each word.
        # if len(words) != len(s):
        #     return False
        
        # # Iterate through the words and s simultaneously.
        # for i in range(len(words)):
        #     # Compare the first character of the current word
        #     # with the character at the corresponding position in s.
        #     if words[i][0] != s[i]:
        #         return False # Mismatch found
            
        # # If all characters match, s is the acronym.
        # return True