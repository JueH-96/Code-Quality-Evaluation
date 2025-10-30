class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        n1 = len(word1)
        n2 = len(word2)
        count = 0
        
        for i in range(n1):
            for j in range(i, n1):
                sub = word1[i:j+1]
                if len(sub) >= n2:
                    
                    sub_counts = {}
                    for char in sub:
                        sub_counts[char] = sub_counts.get(char, 0) + 1
                    
                    word2_counts = {}
                    for char in word2:
                        word2_counts[char] = word2_counts.get(char, 0) + 1
                    
                    valid = True
                    for char, freq in word2_counts.items():
                        if char not in sub_counts or sub_counts[char] < freq:
                            valid = False
                            break
                    if valid:
                        count += 1
        return count