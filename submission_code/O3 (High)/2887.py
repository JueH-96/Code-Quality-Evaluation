class Solution:
    def sortVowels(self, s: str) -> str:
        # Set containing all vowels (both lowercase and uppercase)
        vowel_set = set("AEIOUaeiou")
        
        # Extract vowels from the string and sort them by ASCII value
        vowels_sorted = sorted([ch for ch in s if ch in vowel_set])
        
        # Pointer to the next vowel to place
        idx = 0
        
        # Build the resulting string
        result_chars = []
        for ch in s:
            if ch in vowel_set:                 # Position originally held a vowel
                result_chars.append(vowels_sorted[idx])
                idx += 1
            else:                               # Consonant; keep as is
                result_chars.append(ch)
        
        return "".join(result_chars)