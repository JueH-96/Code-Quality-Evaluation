class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        # Edge case: if n < 5, it's impossible to have all 5 vowels in a substring
        if n < 5:
            return 0
        
        # Helper arrays to count occurrences of each vowel and consonants via prefix sums
        prefix_a = [0] * (n + 1)
        prefix_e = [0] * (n + 1)
        prefix_i = [0] * (n + 1)
        prefix_o = [0] * (n + 1)
        prefix_u = [0] * (n + 1)
        prefix_consonants = [0] * (n + 1)
        
        vowels = {'a', 'e', 'i', 'o', 'u'}
        
        # Build prefix sums
        for idx in range(n):
            prefix_a[idx + 1] = prefix_a[idx] + (1 if word[idx] == 'a' else 0)
            prefix_e[idx + 1] = prefix_e[idx] + (1 if word[idx] == 'e' else 0)
            prefix_i[idx + 1] = prefix_i[idx] + (1 if word[idx] == 'i' else 0)
            prefix_o[idx + 1] = prefix_o[idx] + (1 if word[idx] == 'o' else 0)
            prefix_u[idx + 1] = prefix_u[idx] + (1 if word[idx] == 'u' else 0)
            prefix_consonants[idx + 1] = prefix_consonants[idx] + (0 if word[idx] in vowels else 1)
        
        count_substrings = 0
        
        # Check all possible substrings of length >= 5
        for start in range(n):
            for end in range(start + 4, n):  # at least length 5 to fit all vowels
                a_count = prefix_a[end + 1] - prefix_a[start]
                e_count = prefix_e[end + 1] - prefix_e[start]
                i_count = prefix_i[end + 1] - prefix_i[start]
                o_count = prefix_o[end + 1] - prefix_o[start]
                u_count = prefix_u[end + 1] - prefix_u[start]
                consonant_count = prefix_consonants[end + 1] - prefix_consonants[start]
                
                # Check if all vowels appear at least once and consonants == k
                if (a_count > 0 and e_count > 0 and i_count > 0
                    and o_count > 0 and u_count > 0 and consonant_count == k):
                    count_substrings += 1
        
        return count_substrings