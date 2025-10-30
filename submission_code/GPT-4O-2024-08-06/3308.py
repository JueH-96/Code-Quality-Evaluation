class Solution:
    def lastNonEmptyString(self, s: str) -> str:
        # Convert the string to a list for easier manipulation
        s_list = list(s)
        
        # Keep track of the last non-empty string
        last_non_empty = s
        
        # Perform the operation until the string becomes empty
        while s_list:
            # Create a set to track which characters have been removed in this iteration
            removed_chars = set()
            
            # Iterate over the string and remove the first occurrence of each character
            i = 0
            while i < len(s_list):
                char = s_list[i]
                if char not in removed_chars:
                    removed_chars.add(char)
                    s_list.pop(i)
                else:
                    i += 1
            
            # If the list is not empty, update the last non-empty string
            if s_list:
                last_non_empty = ''.join(s_list)
        
        return last_non_empty