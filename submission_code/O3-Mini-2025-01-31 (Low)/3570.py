class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels_set = {'a', 'e', 'i', 'o', 'u'}
        n = len(word)
        count = 0
        
        # Check for each starting index of substring.
        for i in range(n):
            current_vowels = set()
            consonant_count = 0
            # For each potential ending index j
            for j in range(i, n):
                ch = word[j]
                # Check if vowel or consonant
                if ch in vowels_set:
                    current_vowels.add(ch)
                else:
                    consonant_count += 1
                # Only valid if substring length is at least 5, but that's naturally enforced
                if current_vowels == vowels_set and consonant_count == k:
                    count += 1
                # Additional early break: if consonant_count > k, 
                # going further can only add more consonants. So break.
                if consonant_count > k:
                    break
        return count

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.countOfSubstrings("aeioqq", 1))      # Expected output: 0
    print(sol.countOfSubstrings("aeiou", 0))       # Expected output: 1
    print(sol.countOfSubstrings("ieaouqqieaouqq", 1)) # Expected output: 3