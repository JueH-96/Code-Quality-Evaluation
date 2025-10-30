class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        count = 0
        n = len(words)
        for i in range(n):
            for j in range(i + 1, n):
                if self.isPrefixAndSuffix(words[i], words[j]):
                    count += 1
        return count

    def isPrefixAndSuffix(self, str1: str, str2: str) -> bool:
        if len(str1) > len(str2):
            return False
        
        prefix = str2[:len(str1)]
        suffix = str2[len(str2) - len(str1):]
        
        return prefix == str1 and suffix == str1