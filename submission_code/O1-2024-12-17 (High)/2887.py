class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = set("aeiouAEIOU")
        # Collect all vowels in a list
        vowel_chars = [ch for ch in s if ch in vowels]
        # Sort them by their ASCII values
        vowel_chars.sort()
        
        # Build the resulting string
        result = []
        idx = 0
        for ch in s:
            if ch in vowels:
                # Insert the next sorted vowel
                result.append(vowel_chars[idx])
                idx += 1
            else:
                # Keep the consonant in its original position
                result.append(ch)
        
        return "".join(result)