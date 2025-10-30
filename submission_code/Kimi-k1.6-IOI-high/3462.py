class Solution:
    def doesAliceWin(self, s: str) -> bool:
        vowels = {'a', 'e', 'i', 'o', 'u'}
        n = len(s)
        prefix = [0] * (n + 1)
        vowel_count = 0
        
        # Compute prefix sums modulo 2 for vowels
        for i in range(n):
            prefix[i + 1] = prefix[i] + (1 if s[i] in vowels else 0)
            prefix[i + 1] %= 2
        
        # Check if there are any vowels
        total_vowels = sum(1 for c in s if c in vowels)
        if total_vowels == 0:
            return False
        
        # Check if entire string has odd vowels
        if prefix[-1] == 1:
            return True
        
        # Check if there's at least one position where prefix changes parity
        # This means there's a substring with odd number of vowels
        has_odd = False
        for i in range(n + 1):
            if prefix[i] != prefix[0]:
                has_odd = True
                break
        
        return has_odd