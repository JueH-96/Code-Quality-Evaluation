class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        count = 0
        for i in range(n):
            vowel_set = set()
            cons_count = 0
            for j in range(i, n):
                if word[j] in 'aeiou':
                    vowel_set.add(word[j])
                else:
                    cons_count += 1
                if len(vowel_set) == 5 and cons_count == k:
                    count += 1
        return count