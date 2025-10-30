class Solution:
    def maxOperations(self, s: str) -> int:
        s_list = list(s)  # Convert to list for easier manipulation
        operations = 0
        
        i = 0
        while i < len(s_list) - 1:
            if s_list[i] == '1' and s_list[i + 1] == '0':
                # Found a "10" pattern
                s_list[i] = '0'  # Replace the '1' with '0'
                
                # Find the next '1' if any
                j = i + 2
                while j < len(s_list) and s_list[j] == '0':
                    j += 1
                
                if j < len(s_list):  # If there's a '1' to the right
                    s_list[j-1] = '1'  # Place the '1' just before the next '1'
                else:  # If there's no '1' to the right
                    s_list[-1] = '1'  # Place the '1' at the end
                
                operations += 1
                i = 0  # Reset search to beginning since string changed
            else:
                i += 1
        
        return operations