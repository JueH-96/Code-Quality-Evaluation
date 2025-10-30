class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        def distance(char1, char2):
            diff = ord(char1) - ord(char2)
            return min(diff, 26 - abs(diff))
        
        result = list(s)
        for i in range(len(s) - 1, -1, -1):
            min_dist = distance(s[i], 'a')
            if k - min_dist >= 0: 
                k -= min_dist
                result[i] = 'a'
            else:
                # Calculate the new character that results in distance <= k
                new_char_code = ord(s[i]) - k
                # Ensure new_char_code falls within 'a' to 'z'
                while new_char_code > ord('z'):
                    new_char_code -= 26
                # Update the character in result list
                result[i] = chr(new_char_code)
                break
        return "".join(result)