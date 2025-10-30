class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        n = len(s)
        # Convert s to a list of characters for in-place modification.
        # This list will store the characters of our resulting string t.
        t_chars = list(s) 
        
        # Iterate through each character of the string s from left to right.
        for i in range(n):
            current_char_s = s[i]
            
            # Calculate the cost to change current_char_s to 'a'.
            # The distance between two characters in a cyclic alphabet of 26 letters
            # is the minimum of the absolute difference between their ordinal values
            # and (26 - absolute difference).
            linear_dist_to_a = abs(ord(current_char_s) - ord('a'))
            cost_to_a = min(linear_dist_to_a, 26 - linear_dist_to_a)
            
            # Apply the greedy strategy:
            # Try to change the current character to 'a' if we have enough budget.
            if k >= cost_to_a:
                # If we have enough budget (k) to change s[i] to 'a',
                # we do it because 'a' is the lexicographically smallest character.
                t_chars[i] = 'a'
                k -= cost_to_a
            else:
                # If we don't have enough budget to change s[i] to 'a',
                # we use all the remaining budget 'k' to move s[i] as much as possible
                # towards 'a' to make it lexicographically smallest.
                # Moving towards 'a' means decreasing the character's ASCII value.
                # For example, if s[i] is 'x' and k is 1, the new char will be 'w' (ord('x') - 1).
                # The distance between 'x' and 'w' is 1, which equals the remaining k.
                t_chars[i] = chr(ord(current_char_s) - k)
                k = 0 # All remaining budget is now used up
                
                # Since k is 0, no further characters can be changed.
                # The rest of the characters in t_chars (from i+1 onwards) are already
                # the original characters from s (because t_chars was initialized with list(s)),
                # so we can stop processing.
                break 
        
        # Join the list of characters to form the final string.
        return "".join(t_chars)