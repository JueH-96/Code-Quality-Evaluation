class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        # We want substrings that contain all vowels at least once and exactly k consonants.
        vowels = {'a','e','i','o','u'}
        n = len(word)
        
        # Build prefix counts for each vowel and for consonants
        prefix_a = [0]*(n+1)
        prefix_e = [0]*(n+1)
        prefix_i = [0]*(n+1)
        prefix_o = [0]*(n+1)
        prefix_u = [0]*(n+1)
        prefix_cons = [0]*(n+1)

        for idx, ch in enumerate(word):
            # Copy previous sums
            prefix_a[idx+1] = prefix_a[idx]
            prefix_e[idx+1] = prefix_e[idx]
            prefix_i[idx+1] = prefix_i[idx]
            prefix_o[idx+1] = prefix_o[idx]
            prefix_u[idx+1] = prefix_u[idx]
            prefix_cons[idx+1] = prefix_cons[idx]
            
            # Update specific counts
            if ch == 'a':
                prefix_a[idx+1] += 1
            elif ch == 'e':
                prefix_e[idx+1] += 1
            elif ch == 'i':
                prefix_i[idx+1] += 1
            elif ch == 'o':
                prefix_o[idx+1] += 1
            elif ch == 'u':
                prefix_u[idx+1] += 1
            else:
                # It's a consonant
                prefix_cons[idx+1] += 1

        # Count valid substrings
        result = 0
        for start in range(n):
            for end in range(start, n):
                # Count of each vowel in the substring word[start:end+1]
                count_a = prefix_a[end+1] - prefix_a[start]
                count_e = prefix_e[end+1] - prefix_e[start]
                count_i = prefix_i[end+1] - prefix_i[start]
                count_o = prefix_o[end+1] - prefix_o[start]
                count_u = prefix_u[end+1] - prefix_u[start]
                
                # Count of consonants
                cons_count = prefix_cons[end+1] - prefix_cons[start]
                
                # Check if all vowels are at least present once and consonants count is k
                if (count_a > 0 and count_e > 0 and count_i > 0 and
                    count_o > 0 and count_u > 0 and cons_count == k):
                    result += 1
        
        return result