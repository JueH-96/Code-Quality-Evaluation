class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = set("aeiouAEIOU")
        # Extract vowels from s
        vowel_chars = [ch for ch in s if ch in vowels]
        # Sort them by ASCII value
        vowel_chars.sort()
        
        result = []
        # Replace vowels in s with sorted vowels in order
        idx = 0
        for ch in s:
            if ch in vowels:
                result.append(vowel_chars[idx])
                idx += 1
            else:
                result.append(ch)
                
        return "".join(result)