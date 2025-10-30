class Solution:
    def sortVowels(self, s: str) -> str:
        vowels = "aeiouAEIOU"
        vowel_list = []
        result = list(s)
        for i, char in enumerate(s):
            if char in vowels:
                vowel_list.append((i, char))
        vowel_list.sort(key=lambda x: x[1])
        
        k=0
        for i, char in enumerate(result):
            if char in vowels:
                result[i] = vowel_list[k][1]
                k+=1
        return "".join(result)