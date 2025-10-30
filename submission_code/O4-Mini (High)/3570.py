class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        # Prepare quick vowel checks and bit indices
        is_vowel = [False] * 26
        vowel_to_bit = [0] * 26
        for ch, b in [('a', 0), ('e', 1), ('i', 2), ('o', 3), ('u', 4)]:
            idx = ord(ch) - ord('a')
            is_vowel[idx] = True
            vowel_to_bit[idx] = b

        FULL_MASK = (1 << 5) - 1  # 0b11111
        result = 0

        # Try every starting position i
        for i in range(n):
            mask = 0      # which vowels we've seen in word[i..j]
            cons = 0      # count of consonants in word[i..j]
            # extend the substring to j
            for j in range(i, n):
                idx = ord(word[j]) - ord('a')
                if is_vowel[idx]:
                    mask |= 1 << vowel_to_bit[idx]
                else:
                    cons += 1
                    if cons > k:
                        # once we exceed k consonants, extending further won't help
                        break
                # check if we now have exactly k consonants and all 5 vowels
                if cons == k and mask == FULL_MASK:
                    result += 1

        return result