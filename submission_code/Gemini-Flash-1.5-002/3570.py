class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = "aeiou"
        count = 0
        for i in range(len(word)):
            for j in range(i, len(word)):
                substring = word[i:j+1]
                vowel_present = True
                consonant_count = 0
                vowel_count = 0
                for char in substring:
                    if char in vowels:
                        vowel_count +=1
                    else:
                        consonant_count += 1

                if vowel_count == 5 and consonant_count == k:
                    count += 1
        return count