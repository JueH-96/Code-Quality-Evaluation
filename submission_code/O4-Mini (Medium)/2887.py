class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = set("aeiouAEIOU")
        # Extract all vowels in s
        vowel_chars = [ch for ch in s if ch in vowels]
        # Sort them by ASCII value (uppercase come before lowercase)
        vowel_chars.sort()
        
        # Rebuild the string, replacing vowels in order with sorted ones
        res = []
        vi = 0
        for ch in s:
            if ch in vowels:
                res.append(vowel_chars[vi])
                vi += 1
            else:
                res.append(ch)
        return "".join(res)