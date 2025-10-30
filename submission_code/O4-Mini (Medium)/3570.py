class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        # Map each vowel to a bit
        vowel_bit = {'a':1, 'e':2, 'i':4, 'o':8, 'u':16}
        FULL = 1|2|4|8|16  # 31, all vowels seen
        
        ans = 0
        for i in range(n):
            mask = 0        # which vowels we've seen
            cons = 0        # count of consonants
            for j in range(i, n):
                c = word[j]
                if c in vowel_bit:
                    mask |= vowel_bit[c]
                else:
                    cons += 1
                if cons > k:
                    break
                # if we've used exactly k consonants and have all vowels
                if cons == k and mask == FULL:
                    ans += 1
        return ans