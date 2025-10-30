class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        
        # Count the occurrences of each character
        char_count = [0] * 26
        for char in s:
            char_count[ord(char) - ord('a')] += 1
        
        # Perform t transformations
        for _ in range(t):
            new_count = [0] * 26
            for i in range(26):
                if i == 25:  # 'z'
                    new_count[0] += char_count[i]  # 'a'
                    new_count[1] += char_count[i]  # 'b'
                else:
                    new_count[i + 1] += char_count[i]
            char_count = new_count
        
        # Calculate the total length
        total_length = sum(char_count) % MOD
        
        return total_length