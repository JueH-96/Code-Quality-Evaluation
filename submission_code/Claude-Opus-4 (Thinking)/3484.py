class Solution:
    def getSmallestString(self, s: str) -> str:
        s_list = list(s)
        n = len(s)
        
        for i in range(n - 1):
            # Check if adjacent digits have same parity
            if int(s_list[i]) % 2 == int(s_list[i+1]) % 2:
                # Check if swapping would make it lexicographically smaller
                if s_list[i] > s_list[i+1]:
                    # Swap
                    s_list[i], s_list[i+1] = s_list[i+1], s_list[i]
                    # Return immediately after first swap
                    return ''.join(s_list)
        
        # No beneficial swap found
        return s