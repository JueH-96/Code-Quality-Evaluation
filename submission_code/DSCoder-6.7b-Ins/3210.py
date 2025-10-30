class Solution:
    def beautifulSubstrings(self, s: str, k: int) -> int:
        vowels = 'aeiou'
        n = len(s)
        count = 0
        for i in range(n-k+1):
            sub = s[i:i+k]
            v = sum(c in vowels for c in sub)
            c = k - v
            if v == c and (v * c) % k == 0:
                count += 1
        return count