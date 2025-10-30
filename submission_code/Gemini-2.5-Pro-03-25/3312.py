class Solution:
    def countKeyChanges(self, s: str) -> int:
        """
        Counts the number of times the key had to be changed while typing the string s.
        A key change occurs when a typed character corresponds to a different key 
        than the previously typed character, ignoring case differences (e.g., 'a' and 'A' 
        are considered the same key).

        Args:
            s: The input string typed by the user. Consists of only upper case 
               and lower case English letters. Length is between 1 and 100 according
               to constraints.

        Returns:
            The total number of times the key was changed.
        """
        
        # Initialize the count of key changes
        changes = 0
        
        # Get the length of the string
        n = len(s)
            
        # Iterate through the string starting from the second character (index 1).
        # We compare each character s[i] with the previous character s[i-1].
        # The loop runs only if n > 1. If n is 1, range(1, 1) is empty, 
        # and the function correctly returns 0 changes.
        for i in range(1, n):
            # We use .lower() to make the comparison case-insensitive, 
            # treating 'a' and 'A' as the same key.
            if s[i].lower() != s[i-1].lower():
                # If the lowercase versions of the adjacent characters are different,
                # it means a different key was pressed, so we increment the counter.
                changes += 1
                
        # Return the total count of key changes detected.
        return changes

# Example of using the zip-based alternative (also correct and slightly more concise):
# class Solution:
#     def countKeyChanges(self, s: str) -> int:
#         """
#         Counts the number of times the key had to be changed while typing the string s,
#         ignoring case. Uses zip for comparing adjacent elements.
#         """
#         # zip(s, s[1:]) creates an iterator of adjacent pairs (s[0], s[1]), (s[1], s[2]), ...
#         # For each pair (c1, c2), c1.lower() != c2.lower() checks if the keys are different (case-insensitive).
#         # This results in True (1) or False (0).
#         # sum() adds up all the instances where the keys were different (True/1).
#         # Handles strings of length 0 or 1 correctly as zip yields no pairs.
#         return sum(c1.lower() != c2.lower() for c1, c2 in zip(s, s[1:]))