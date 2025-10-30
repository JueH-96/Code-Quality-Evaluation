class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        
        # Count the number of each character in the string
        count = [0] * 26
        for char in s:
            count[ord(char) - ord('a')] += 1
        
        # Apply transformations
        for _ in range(t):
            # New count array after transformation
            new_count = [0] * 26
            
            # For each character from 'a' to 'y'
            for i in range(25):
                new_count[i + 1] = count[i]
            
            # Special case for 'z'
            new_count[0] = (new_count[0] + count[25]) % MOD  # 'z' to 'ab' contributes to 'a'
            new_count[1] = (new_count[1] + count[25]) % MOD  # 'z' to 'ab' contributes to 'b'
            
            # Update count array
            count = new_count
        
        # Calculate the total length of the string after t transformations
        total_length = sum(count) % MOD
        
        return total_length