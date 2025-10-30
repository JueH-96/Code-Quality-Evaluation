class Solution:
    def minAnagramLength(self, s: str) -> int:
        char_count = {}
        for char in s:
            char_count[char] = char_count.get(char, 0) + 1
        
        gcd = char_count[s[0]]
        for count in char_count.values():
            gcd = self.gcd(gcd, count)
        
        return len(s) // gcd
    
    def gcd(self, a: int, b: int) -> int:
        while b:
            a, b = b, a % b
        return a