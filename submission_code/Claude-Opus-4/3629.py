class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        
        # Count initial frequency of each character
        count = [0] * 26
        for char in s:
            count[ord(char) - ord('a')] += 1
        
        # Perform t transformations
        for _ in range(t):
            new_count = [0] * 26
            
            # Each character transforms to the next one
            for i in range(26):
                if count[i] > 0:
                    if i == 25:  # 'z' transforms to "ab"
                        new_count[0] = (new_count[0] + count[i]) % MOD
                        new_count[1] = (new_count[1] + count[i]) % MOD
                    else:  # Other characters transform to the next character
                        new_count[i + 1] = (new_count[i + 1] + count[i]) % MOD
            
            count = new_count
        
        # Calculate total length
        total_length = sum(count) % MOD
        return total_length