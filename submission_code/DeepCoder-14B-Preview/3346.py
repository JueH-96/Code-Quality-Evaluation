class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        s_list = list(s)
        remaining = k
        n = len(s_list)
        
        for i in range(n):
            current_char = s_list[i]
            for c in 'abcdefghijklmnopqrstuvwxyz':
                # Calculate the minimal distance between current_char and c
                d = abs(ord(c) - ord(current_char))
                min_d = min(d, 26 - d)
                if min_d <= remaining:
                    s_list[i] = c
                    remaining -= min_d
                    break  # Move to the next character after finding the smallest possible c
        return ''.join(s_list)