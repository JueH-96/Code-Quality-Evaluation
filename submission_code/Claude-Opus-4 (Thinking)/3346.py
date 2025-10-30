class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        result = []
        
        for char in s:
            # Find the lexicographically smallest character we can change to
            for target_ord in range(ord('a'), ord('z') + 1):
                target_char = chr(target_ord)
                
                # Calculate the cyclic distance
                diff = abs(target_ord - ord(char))
                dist = min(diff, 26 - diff)
                
                if dist <= k:
                    result.append(target_char)
                    k -= dist
                    break
        
        return ''.join(result)