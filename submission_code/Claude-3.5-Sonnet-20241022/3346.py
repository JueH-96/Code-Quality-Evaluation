class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        def get_dist(c1, c2):
            d1 = abs(ord(c1) - ord(c2))
            d2 = 26 - d1
            return min(d1, d2)
            
        def get_total_dist(s1, s2):
            return sum(get_dist(c1, c2) for c1, c2 in zip(s1, s2))
        
        n = len(s)
        result = list(s)
        
        # Try to make each character as small as possible from left to right
        for i in range(n):
            original = result[i]
            # Try each possible character from 'a' up to original
            for c in range(ord('a'), ord(original) + 1):
                result[i] = chr(c)
                # Check if total distance is still within k
                if get_total_dist(s, ''.join(result)) <= k:
                    break
            else:
                # If no character works, restore original
                result[i] = original
                
        return ''.join(result)