class Solution:
    def sortVowels(self, s: str) -> str:
        # Define the set of vowels (both lowercase and uppercase)
        vowel_set = set("aeiouAEIOU")
        
        # Extract and sort all vowels by their ASCII values
        vowels = [ch for ch in s if ch in vowel_set]
        vowels.sort()
        
        # Reconstruct the string, replacing each vowel with the next from the sorted list
        res = []
        vi = 0
        for ch in s:
            if ch in vowel_set:
                res.append(vowels[vi])
                vi += 1
            else:
                res.append(ch)
        
        return "".join(res)