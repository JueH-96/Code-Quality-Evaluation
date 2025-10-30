class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        def char_distance(c1, c2):
            # Calculate minimum cyclic distance between two characters
            diff = abs(ord(c1) - ord(c2))
            return min(diff, 26 - diff)
        
        result = list(s)
        remaining_k = k
        
        for i in range(len(s)):
            current_char = s[i]
            
            # Try to change to 'a' first (lexicographically smallest)
            cost_to_a = char_distance(current_char, 'a')
            
            if cost_to_a <= remaining_k:
                # We can afford to change to 'a'
                result[i] = 'a'
                remaining_k -= cost_to_a
            else:
                # We can't afford 'a', find the smallest character we can afford
                # Try each character from 'a' to 'z'
                for target_ord in range(ord('a'), ord('z') + 1):
                    target_char = chr(target_ord)
                    cost = char_distance(current_char, target_char)
                    
                    if cost <= remaining_k:
                        result[i] = target_char
                        remaining_k -= cost
                        break
        
        return ''.join(result)