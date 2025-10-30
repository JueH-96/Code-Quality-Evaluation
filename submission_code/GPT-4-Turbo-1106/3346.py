class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        def distance(c1, c2):
            return min(abs(ord(c1) - ord(c2)), 26 - abs(ord(c1) - ord(c2)))
        
        t = list(s)
        for i in range(len(s)):
            if k == 0:
                break
            # Calculate the distance to 'a'
            dist_to_a = distance(s[i], 'a')
            # If we can afford to change the current character to 'a'
            if dist_to_a <= k:
                t[i] = 'a'
                k -= dist_to_a
            else:
                # Find the character we can change to within the remaining distance k
                if s[i] <= 'm':
                    t[i] = chr(ord(s[i]) - k)
                else:
                    t[i] = chr(ord(s[i]) + k)
                k = 0
        return ''.join(t)