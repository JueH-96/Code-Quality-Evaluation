class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        from collections import Counter
        count = 0
        n1 = len(word1)
        n2 = len(word2)
        if n2 > n1:
            return 0
        
        word2_count = Counter(word2)
        
        for i in range(n1):
            for j in range(i,n1):
                substring = word1[i:j+1]
                substring_count = Counter(substring)
                
                valid = True
                for char, freq in word2_count.items():
                    if substring_count[char] < freq:
                        valid = False
                        break
                if valid:
                    count +=1
        return count