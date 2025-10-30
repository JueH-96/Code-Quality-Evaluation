class Solution:
    def smallestString(self, s: str) -> str:
        # Convert the string into a list for in-place modifications
        s_list = list(s)
        n = len(s_list)
        found = False
        
        # Iterate over the string to find the first non-'a' character.
        # Once found, start reducing each consecutive non-'a' character.
        for i in range(n):
            if not found and s_list[i] != 'a':
                found = True
                s_list[i] = chr(ord(s_list[i]) - 1)
            elif found and s_list[i] != 'a':
                s_list[i] = chr(ord(s_list[i]) - 1)
            elif found:
                # We encountered an 'a' after starting the decrement process.
                # So we stop the operation.
                break
        
        # If no character was reduced (i.e., s consists entirely of 'a's),
        # we must perform an operation, so change the last 'a' to 'z'.
        if not found:
            s_list[-1] = 'z'
        
        return "".join(s_list)