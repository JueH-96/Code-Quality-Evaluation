class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        
        # Count the occurrences of each character in the original string
        counts = [0] * 26
        for char in s:
            counts[ord(char) - ord('a')] += 1
        
        # Perform transformations
        for _ in range(t):
            new_counts = [0] * 26
            
            # Handle 'z' transformation - becomes "ab"
            new_counts[0] = (new_counts[0] + counts[25]) % MOD  # 'z' -> 'a'
            new_counts[1] = (new_counts[1] + counts[25]) % MOD  # 'z' -> 'b'
            
            # Handle other transformations - each becomes next character
            for i in range(25):
                new_counts[i+1] = (new_counts[i+1] + counts[i]) % MOD
            
            counts = new_counts
        
        # Calculate the total length of the string after t transformations
        length = 0
        for count in counts:
            length = (length + count) % MOD
        
        return length