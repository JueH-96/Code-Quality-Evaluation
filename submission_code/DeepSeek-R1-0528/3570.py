class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(word)
        res = 0
        for i in range(n):
            seen = set()
            cons = 0
            for j in range(i, n):
                ch = word[j]
                if ch in vowels:
                    seen.add(ch)
                else:
                    cons += 1
                if cons > k:
                    break
                if j - i + 1 < 5:
                    continue
                if cons == k and len(seen) == 5:
                    res += 1
        return res