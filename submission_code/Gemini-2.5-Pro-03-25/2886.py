import collections # Importing collections just in case, although not used in the final chosen approach

class Solution:
    def finalString(self, s: str) -> str:
        """
        Simulates typing the string s on a faulty keyboard where typing 'i' reverses the text currently typed.

        Args:
            s: The input string to type, character by character. 
               Constraints: 1 <= s.length <= 100, s consists of lowercase English letters, s[0] != 'i'.

        Returns:
            The final string that appears on the screen after typing all characters from s.
        """
        
        # We can simulate the process of building the string. Using a list of characters
        # is often more efficient for modifications like appending compared to 
        # repeatedly creating new strings via concatenation in Python.
        result_list = []
        
        # Iterate through each character of the input string s
        for char in s:
            # Check if the typed character is 'i'
            if char == 'i':
                # If the character is 'i', the current text on the screen gets reversed.
                # We simulate this by reversing the list of characters in-place.
                # The time complexity of list.reverse() is O(k), where k is the current length of the list.
                result_list.reverse()
            else:
                # If the character is not 'i', it's appended to the end of the current text.
                # We simulate this by appending the character to our list.
                # The time complexity of list.append() is O(1) amortized.
                result_list.append(char)
                
        # After processing all characters in the input string s,
        # the list `result_list` contains the characters of the final string in order.
        # We join the characters in the list to form the final string.
        # The time complexity of "".join(list) is O(L), where L is the final length of the string.
        return "".join(result_list)