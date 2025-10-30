class Solution:
    def countPrefixAndSuffix(self, str1, str2):
        n1 = len(str1)
        n2 = len(str2)
        if n1 > n2:
            return False

        for i in range(n1):
            if str1[i] != str2[i]:
                return False

        for i in range(n1):
            if str1[i] != str2[n2 - n1 + i]:
                return False
        
        return True

    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        count = 0
        n = len(words)
        for i in range(n):
            for j in range(i + 1, n):
                if self.countPrefixAndSuffix(words[i], words[j]):
                    count += 1
        return count