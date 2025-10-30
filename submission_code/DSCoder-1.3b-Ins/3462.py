class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = 'aeiou'
        alice_vowel_count = sum(s[i] in vowels for i in range(len(s)))
        bob_vowel_count = len(s) - alice_vowel_count
        if alice_vowel_count % 2 == 0:
            return bob_vowel_count % 2 == 0
        else:
            return alice_vowel_count % 2 == 0