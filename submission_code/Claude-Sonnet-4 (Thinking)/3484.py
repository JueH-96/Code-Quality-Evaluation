class Solution:
    def getSmallestString(self, s: str) -> str:
        s_list = list(s)
        
        for i in range(len(s) - 1):
            # Check if adjacent digits have same parity
            if int(s[i]) % 2 == int(s[i+1]) % 2:
                # Check if swapping would make it lexicographically smaller
                if s[i+1] < s[i]:
                    # Swap and return
                    s_list[i], s_list[i+1] = s_list[i+1], s_list[i]
                    return ''.join(s_list)
        
        # No beneficial swap found
        return s