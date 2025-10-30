class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        vowels = {'a', 'e', 'i', 'o', 'u'}
        # Build a prefix sum array for the number of vowels.
        prefix = [0] * (n + 1)
        for i, char in enumerate(s):
            prefix[i + 1] = prefix[i] + (1 if char in vowels else 0)
        
        count_beautiful = 0
        # Only consider substrings of even length since vowels must equal consonants.
        for i in range(n):
            for j in range(i + 2, n + 1, 2):
                # Number of vowels in the substring s[i:j]
                num_vowels = prefix[j] - prefix[i]
                length = j - i
                # For the substring to have equal vowels and consonants,
                # the number of vowels must equal half the substring length.
                if num_vowels == length // 2:
                    # Then, the product is num_vowels * num_vowels.
                    if (num_vowels * num_vowels) % k == 0:
                        count_beautiful += 1
        return count_beautiful