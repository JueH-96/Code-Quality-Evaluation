class Solution:
    def getSmallestString(self, s: str, k: int) -> str:
        n = len(s)
        t = list(s)  # Convert string to a list for mutability
        
        for i in range(n):
            if k <= 0:
                break
            
            # Calculate the distance to 'a'
            dist_to_a = min((ord(s[i]) - ord('a')) % 26, (ord('z') - ord(s[i]) + 1) % 26)
            
            # If we can change s[i] to 'a' within the remaining k
            if dist_to_a <= k:
                t[i] = 'a'  # Change to 'a'
                k -= dist_to_a  # Reduce k by the distance used
            else:
                # If we can't change to 'a', we can change it to the closest character
                # that uses up the remaining k
                if k > 0:
                    # Find the new character we can change to
                    new_char = chr(ord(s[i]) - k) if (ord(s[i]) - k) >= ord('a') else chr(ord(s[i]) + k)
                    t[i] = new_char
                    k = 0  # All k is used up
        
        return ''.join(t)