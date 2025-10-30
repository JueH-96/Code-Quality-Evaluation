class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = set('aeiou')
        n = len(s)
        prefix_diff = [0] * (n + 1)
        
        for i in range(n):
            prefix_diff[i + 1] = prefix_diff[i] + (1 if s[i] in vowels else -1)
        
        count = 0
        for i in range(n):
            for j in range(i, n):
                length = j - i + 1
                vowels_count = (prefix_diff[j + 1] - prefix_diff[i] + length) // 2
                consonants_count = length - vowels_count
                
                if vowels_count == consonants_count and (vowels_count * consonants_count) % k == 0:
                    count += 1
        
        return count