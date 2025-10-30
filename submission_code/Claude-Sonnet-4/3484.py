class Solution:
    def getSmallestString(self, s: str) -> str:
        s_list = list(s)
        
        for i in range(len(s) - 1):
            # Check if adjacent digits have same parity
            digit1 = int(s_list[i])
            digit2 = int(s_list[i + 1])
            
            # Same parity means both odd or both even
            if digit1 % 2 == digit2 % 2:
                # If swapping makes it smaller lexicographically
                if digit2 < digit1:
                    # Swap and return immediately (we want the first beneficial swap)
                    s_list[i], s_list[i + 1] = s_list[i + 1], s_list[i]
                    return ''.join(s_list)
        
        # No beneficial swap found
        return s