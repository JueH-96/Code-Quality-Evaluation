class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowel = set('aeiou')
        n = len(word)
        ans = 0
        vowel_cnt, conso_cnt = 0, 0 
        vowels = 0
        indices = [0] * 6
        for idx, char in enumerate(word):
            if char in vowel: 
                vowel_cnt += 1
                indices[ord(char) - ord('a')] = idx + 1
                vowels |= (1 << (ord(char) - ord('a')))
            else:
                conso_cnt += 1

            if vowel_cnt == 5 and conso_cnt >= k:
                ans += min(indices) + 1 - conso_cnt + k
            if idx >= 4 and word[idx-4] in vowel: 
                vowel_cnt -= 1
                vowels &= ~(1 << (ord(word[idx-4]) - ord('a')))
            if idx >= n - k - 1 and word[idx-(n-k-1)] not in vowel: 
                conso_cnt -= 1
                
        return ans