class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        n = len(word)
        ans = 0
        for i in range(n):
            for j in range(i, n):
                sub = word[i:j+1]
                vowels = {'a': 0, 'e': 0, 'i': 0, 'o': 0, 'u': 0}
                consonants = 0
                for char in sub:
                    if char in vowels:
                        vowels[char] += 1
                    else:
                        consonants += 1
                
                if all(vowels.values()) and consonants == k:
                    ans += 1
        return ans