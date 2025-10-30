class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(word)
        prefix_consonant = [0] * (n + 1)
        prefix_vowels = {v: [0] * (n + 1) for v in vowels}
        
        for i in range(n):
            c = word[i]
            # Update consonant prefix
            if c not in vowels:
                prefix_consonant[i + 1] = prefix_consonant[i] + 1
            else:
                prefix_consonant[i + 1] = prefix_consonant[i]
            # Update each vowel's prefix
            for v in vowels:
                if c == v:
                    prefix_vowels[v][i + 1] = prefix_vowels[v][i] + 1
                else:
                    prefix_vowels[v][i + 1] = prefix_vowels[v][i]
        
        count = 0
        # Iterate over all possible substrings
        for start in range(n):
            for end in range(start, n):
                # Calculate consonant count in the substring [start, end]
                current_consonant = prefix_consonant[end + 1] - prefix_consonant[start]
                if current_consonant != k:
                    continue
                # Check if all vowels are present
                all_present = True
                for v in vowels:
                    current_v_count = prefix_vowels[v][end + 1] - prefix_vowels[v][start]
                    if current_v_count < 1:
                        all_present = False
                        break
                if all_present:
                    count += 1
        return count