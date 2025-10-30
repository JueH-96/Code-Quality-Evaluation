class Solution:
    """
    Implements the solution to the clearDigits problem.
    """
    def clearDigits(self, s: str) -> str:
        """
        Removes digits and preceding non-digits based on the specified rule.

        The operation described is: Delete the first digit and the closest 
        non-digit character to its left. Repeat this process until no digits remain.

        This implementation simulates this process efficiently using a stack-like 
        approach with a list. We iterate through the input string `s`:
        - If the current character is a letter (non-digit), we append it to our result list.
        - If the current character is a digit, it signifies that an operation needs 
          to occur. This operation removes this digit and the closest non-digit 
          to its left. In our list-based construction, the "closest non-digit 
          to its left" corresponds precisely to the last character that was added 
          to the list (which must be a non-digit). Therefore, we simply pop the 
          last element from the list.

        The problem constraints guarantee that whenever we encounter a digit, 
        there will be a non-digit character available in our list to be removed 
        (i.e., the list will not be empty when `pop()` is called due to a digit).

        Args:
          s: The input string containing lowercase English letters and digits. 
             Length is between 1 and 100.

        Returns:
          The resulting string after applying the digit removal operation repeatedly 
          until no digits are left.
        """
        # Use a list to function as a stack to build the final string characters.
        res_list = [] 
        
        # Iterate through each character in the input string.
        for char in s:
            # Check if the character is a digit ('0' through '9').
            if char.isdigit():
                # If it's a digit, remove the last character added to the list.
                # This simulates removing the closest non-digit to the left.
                # The constraints guarantee `res_list` is not empty here.
                if res_list: 
                    res_list.pop() 
            else:
                # If it's a letter (non-digit), append it to the list.
                res_list.append(char)
                
        # After processing all characters, join the remaining characters 
        # in the list to form the final result string.
        return "".join(res_list)