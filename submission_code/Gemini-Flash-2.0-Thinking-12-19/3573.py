class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        word2_counts = {}
        for char in word2:
            word2_counts[char] = word2_counts.get(char, 0) + 1
        
        valid_substring_count = 0
        n1 = len(word1)
        n2 = len(word2)
        
        for i in range(n1):
            for j in range(i, n1):
                substring = word1[i:j+1]
                if len(substring) < n2:
                    continue
                
                substring_counts = {}
                for char in substring:
                    substring_counts[char] = substring_counts.get(char, 0) + 1
                    
                is_valid = True
                for char, count in word2_counts.items():
                    if char not in substring_counts or substring_counts[char] < count:
                        is_valid = False
                        break
                if is_valid:
                    valid_substring_count += 1
                    
        return valid_substring_count