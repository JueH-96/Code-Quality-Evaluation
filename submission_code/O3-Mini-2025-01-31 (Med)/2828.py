class Solution:
    def smallestString(self, s: str) -> str:
        # Convert string to list for mutability.
        s_list = list(s)
        n = len(s_list)
        
        # Flag to indicate if we performed the transformation.
        transformed = False
        
        # Find first character not equal to 'a' to start the substring operation.
        for i in range(n):
            if s_list[i] != 'a':
                # From this point, subtract one (wrap 'a' -> 'z') until we hit an 'a'
                while i < n and s_list[i] != 'a':
                    # Decrement letter by one: 'b'->'a', ..., 'z'->'y'
                    s_list[i] = chr(ord(s_list[i]) - 1)
                    i += 1
                transformed = True
                break
        
        # If the string was all 'a's, then we must choose some non-empty substring.
        # The best move is to change the last character from 'a' to 'z'.
        if not transformed:
            s_list[-1] = 'z'
        
        return "".join(s_list)