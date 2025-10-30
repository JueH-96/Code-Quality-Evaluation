class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = []
        consonants = []
        result = []
        
        for char in s:
            if char.lower() in 'aeiou':
                vowels.append(char)
            else:
                consonants.append(char)
        
        vowels.sort()
        
        i = 0
        j = 0
        for char in s:
            if char.lower() in 'aeiou':
                result.append(vowels[i])
                i += 1
            else:
                result.append(char)
        
        return ''.join(result)