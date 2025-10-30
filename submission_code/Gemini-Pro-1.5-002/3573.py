class Solution:
    def validSubstringCount(self, word1: str, word2: str) -> int:
        n1 = len(word1)
        n2 = len(word2)
        ans = 0
        for i in range(n1):
            for j in range(i, n1):
                sub = word1[i:j+1]
                n = len(sub)
                if n < n2:
                    continue
                count1 = [0] * 26
                count2 = [0] * 26
                for char in sub:
                    count1[ord(char) - ord('a')] += 1
                for char in word2:
                    count2[ord(char) - ord('a')] += 1
                
                valid = True
                for k in range(26):
                    if count1[k] < count2[k]:
                        valid = False
                        break
                if valid:
                    ans += 1
        return ans