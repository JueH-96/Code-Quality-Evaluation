class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        count = 0
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                if self.isPrefixAndSuffix(words[i], words[j]):
                    count += 1
        return count

    def isPrefixAndSuffix(self, str1: str, str2: str) -> bool:
        if str1 == str2[:len(str1)] and str2 == str1[:len(str2)]:
            return True
        else:
            return False