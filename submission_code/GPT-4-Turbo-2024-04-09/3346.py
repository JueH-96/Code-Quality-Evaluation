class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        def min_distance(c1, c2):
            return min(abs(ord(c1) - ord(c2)), 26 - abs(ord(c1) - ord(c2)))
        
        n = len(s)
        t = list(s)
        
        for i in range(n):
            if k == 0:
                break
            # Calculate the minimum distance to 'a' from s[i]
            current_char = s[i]
            dist_to_a = min_distance(current_char, 'a')
            
            if dist_to_a <= k:
                # If we can afford to change current_char to 'a'
                t[i] = 'a'
                k -= dist_to_a
            else:
                # If we cannot afford to change to 'a', change as close as possible towards 'a'
                if ord(current_char) >= ord('a'):
                    # Move leftwards towards 'a'
                    new_char = chr(ord(current_char) - k)
                else:
                    # Move rightwards towards 'a'
                    new_char = chr(ord(current_char) + k)
                
                if ord(new_char) < ord('a'):
                    new_char = chr(ord(new_char) + 26)
                elif ord(new_char) > ord('z'):
                    new_char = chr(ord(new_char) - 26)
                
                t[i] = new_char
                k -= min_distance(current_char, new_char)
        
        return ''.join(t)