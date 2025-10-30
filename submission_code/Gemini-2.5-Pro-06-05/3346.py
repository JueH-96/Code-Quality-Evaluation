class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        """
        Finds the lexicographically smallest string t that can be obtained from s
        such that the cyclic distance between s and t is at most k.
        """
        # Convert the string to a list of characters for mutable operations.
        res = list(s)
        
        for i in range(len(res)):
            # If no budget is left, no more changes can be made.
            if k == 0:
                break
            
            char = res[i]
            
            # If the character is already 'a', it's the smallest possible.
            # No change is needed.
            if char == 'a':
                continue
            
            # Calculate the distance to 'a'. The distance moving "up" the alphabet
            # (e.g., d->c->b->a) is `ord(char) - ord('a')`.
            dist = ord(char) - ord('a')
            
            # The cost is the minimum of moving up or moving down (wrapping around).
            cost_to_a = min(dist, 26 - dist)
            
            if k >= cost_to_a:
                # If we have enough budget, change the character to 'a'.
                res[i] = 'a'
                k -= cost_to_a
            else:
                # If not, use the remaining budget k to move the character
                # k steps closer to 'a', making it as small as possible.
                res[i] = chr(ord(char) - k)
                k = 0
                
        return "".join(res)