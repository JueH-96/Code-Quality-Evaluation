class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        def cyclic_distance(a, b):
            return min(abs(ord(a) - ord(b)), 26 - abs(ord(a) - ord(b)))
        
        result = []
        for char in s:
            if k == 0:
                result.append(char)
                continue
            
            # Try to change the character to 'a'
            if cyclic_distance(char, 'a') <= k:
                result.append('a')
                k -= cyclic_distance(char, 'a')
            else:
                # If changing to 'a' exceeds k, find the closest character within k distance
                for new_char in range(ord(char), ord('a') - 1, -1):
                    if cyclic_distance(char, chr(new_char)) <= k:
                        result.append(chr(new_char))
                        k -= cyclic_distance(char, chr(new_char))
                        break
        
        return ''.join(result)