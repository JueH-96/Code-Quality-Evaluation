class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        
        # Find the smallest integer q such that q^2 is divisible by k
        q = 1
        while (q * q) % k != 0:
            q += 1
        
        # Precompute prefix sums for vowels and consonants
        prefix_vowels = [0] * (len(s) + 1)
        prefix_consonants = [0] * (len(s) + 1)
        
        for i, char in enumerate(s):
            prefix_vowels[i+1] = prefix_vowels[i]
            prefix_consonants[i+1] = prefix_consonants[i]
            
            if char in vowels:
                prefix_vowels[i+1] += 1
            else:
                prefix_consonants[i+1] += 1
        
        count = 0
        
        # Dictionary to store (diff, vowel_count % q) and its positions
        # This helps efficiently find positions with equal vowels and consonants
        diff_vmod_positions = {}
        diff_vmod_positions[(0, 0)] = [0]  # Initialize
        
        for i in range(1, len(s) + 1):
            diff = prefix_vowels[i] - prefix_consonants[i]
            vmod = prefix_vowels[i] % q
            
            # Check for positions with the same diff and vmod
            key = (diff, vmod)
            if key in diff_vmod_positions:
                for prev_pos in diff_vmod_positions[key]:
                    v = prefix_vowels[i] - prefix_vowels[prev_pos]
                    c = prefix_consonants[i] - prefix_consonants[prev_pos]
                    
                    if v == c:  # Should be true based on diff, but double-checking
                        count += 1
            
            # Store current position
            if key not in diff_vmod_positions:
                diff_vmod_positions[key] = []
            diff_vmod_positions[key].append(i)
        
        return count