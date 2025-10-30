class Solution:
    def countOfSubstrings(self, word: str, k: int) -> int:
        vowels = set('aeiou')
        n = len(word)
        
        def countAtMostK(max_consonants):
            # Count substrings with all vowels and at most max_consonants consonants
            count = 0
            left = 0
            vowel_count = {}
            consonant_count = 0
            
            for right in range(n):
                char = word[right]
                
                if char in vowels:
                    vowel_count[char] = vowel_count.get(char, 0) + 1
                else:
                    consonant_count += 1
                
                # Shrink window if we have too many consonants
                while consonant_count > max_consonants:
                    left_char = word[left]
                    if left_char in vowels:
                        vowel_count[left_char] -= 1
                        if vowel_count[left_char] == 0:
                            del vowel_count[left_char]
                    else:
                        consonant_count -= 1
                    left += 1
                
                # If we have all vowels, count valid substrings
                if len(vowel_count) == 5:
                    # Find the leftmost position where we still have all vowels
                    temp_left = left
                    temp_vowel_count = vowel_count.copy()
                    temp_consonant_count = consonant_count
                    
                    while temp_left <= right and len(temp_vowel_count) == 5:
                        count += 1
                        
                        left_char = word[temp_left]
                        if left_char in vowels:
                            temp_vowel_count[left_char] -= 1
                            if temp_vowel_count[left_char] == 0:
                                del temp_vowel_count[left_char]
                        else:
                            temp_consonant_count -= 1
                        temp_left += 1
            
            return count
        
        # Count substrings with exactly k consonants = 
        # substrings with at most k consonants - substrings with at most (k-1) consonants
        if k == 0:
            return countAtMostK(0)
        else:
            return countAtMostK(k) - countAtMostK(k - 1)