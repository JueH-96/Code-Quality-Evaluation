class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        beautiful_count = 0
        vowels = {'a', 'e', 'i', 'o', 'u'}

        for i in range(n):
            num_vowels = 0
            num_consonants = 0
            for j in range(i, n):
                char = s[j]
                if 'a' <= char <= 'z':
                    if char in vowels:
                        num_vowels += 1
                    else:
                        num_consonants += 1

                if num_vowels == num_consonants and num_vowels > 0 and (num_vowels * num_consonants) % k == 0:
                    beautiful_count += 1

        return beautiful_count