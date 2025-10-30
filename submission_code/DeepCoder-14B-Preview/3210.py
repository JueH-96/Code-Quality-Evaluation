class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        # Precompute the vowel marks
        vowel_marks = [0] * n
        vowels_set = {'a', 'e', 'i', 'o', 'u'}
        for i in range(n):
            if s[i] in vowels_set:
                vowel_marks[i] = 1
            else:
                vowel_marks[i] = 0
        
        # Compute prefix sums
        prefix = [0] * (n + 1)
        for i in range(n):
            prefix[i+1] = prefix[i] + vowel_marks[i]
        
        count = 0
        for i in range(n):
            for j in range(i, n):
                length = j - i + 1
                if length % 2 != 0:
                    continue
                # Calculate vowels
                vowels = prefix[j+1] - prefix[i]
                consonants = length - vowels
                if vowels != consonants:
                    continue
                # Check if product is divisible by k
                if (vowels * consonants) % k == 0:
                    count += 1
        return count