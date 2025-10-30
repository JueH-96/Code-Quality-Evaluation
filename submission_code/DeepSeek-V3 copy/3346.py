class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        def cyclic_distance(c1, c2):
            # Calculate the minimum distance between two characters in cyclic order
            diff = abs(ord(c1) - ord(c2))
            return min(diff, 26 - diff)
        
        t = list(s)
        remaining_k = k
        for i in range(len(t)):
            # Try to change t[i] to the smallest possible character 'a'
            current_char = t[i]
            target_char = 'a'
            distance_needed = cyclic_distance(current_char, target_char)
            if distance_needed <= remaining_k:
                t[i] = target_char
                remaining_k -= distance_needed
            else:
                # If we can't change to 'a', find the smallest character that can be reached with the remaining k
                # We need to find the smallest character c such that cyclic_distance(current_char, c) <= remaining_k
                # Since we want the lex smallest, we start from 'a' and go up
                for c in range(ord('a'), ord('z') + 1):
                    c_char = chr(c)
                    dist = cyclic_distance(current_char, c_char)
                    if dist <= remaining_k:
                        t[i] = c_char
                        remaining_k -= dist
                        break
        return ''.join(t)