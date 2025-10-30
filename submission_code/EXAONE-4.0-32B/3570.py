class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = "aeiou"
        vowel_map = {ch: idx for idx, ch in enumerate(vowels)}
        n = len(word)
        res = 0
        for start in range(n):
            freq = [0] * 5
            distinct = 0
            cons_count = 0
            for end in range(start, n):
                char = word[end]
                if char in vowel_map:
                    idx = vowel_map[char]
                    if freq[idx] == 0:
                        distinct += 1
                    freq[idx] += 1
                else:
                    cons_count += 1
                
                if cons_count > k:
                    break
                
                if distinct == 5 and cons_count == k:
                    res += 1
                    
        return res