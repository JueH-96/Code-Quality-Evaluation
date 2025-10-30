from typing import List

class Solution:
    def stringSequence(self, target: str) -> List[str]:
        result = []
        current = ""  # current string on screen
        
        # Function to return the next letter in cyclic order
        def next_char(c: str) -> str:
            return 'a' if c == 'z' else chr(ord(c) + 1)
        
        # We process each character of target one by one.
        for ch in target:
            # Step 1: Press Key 1 to append 'a'
            current += 'a'
            result.append(current)
            
            # Step 2: Update the newly appended character to ch using Key 2
            # Only need to update if it is not already ch.
            # We change only the last character.
            while current[-1] != ch:
                # Get the next character (cyclic update)
                updated_char = next_char(current[-1])
                # Replace the last character with its updated version
                current = current[:-1] + updated_char
                result.append(current)
        
        return result

# For testing the solution locally:
if __name__ == "__main__":
    sol = Solution()
    print(sol.stringSequence("abc"))
    # Expected Output: ["a","aa","ab","aba","abb","abc"]

    print(sol.stringSequence("he"))
    # Expected Output: ["a","b","c","d","e","f","g","h","ha","hb","hc","hd","he"]