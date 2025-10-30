class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        vowels_set = {'a', 'e', 'i', 'o', 'u'}
        # Precompute prefix vowel counts
        prefix_vowels = [0] * (n + 1)
        for i in range(n):
            prefix_vowels[i+1] = prefix_vowels[i] + (1 if s[i] in vowels_set else 0)
        
        count = 0
        # brute force enumeration of substrings O(n^2)
        for i in range(n):
            for j in range(i+1, n+1):
                # substring s[i:j]
                # number of vowels in substring = prefix_vowels[j] - prefix_vowels[i]
                vowels = prefix_vowels[j] - prefix_vowels[i]
                length = j - i
                cons = length - vowels
                if vowels == cons and vowels != 0:
                    # check divisibility
                    if (vowels * cons) % k == 0:
                        count += 1
        return count