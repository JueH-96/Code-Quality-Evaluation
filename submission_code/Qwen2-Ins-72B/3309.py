class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        def isPrefixAndSuffix(str1, str2):
            len1, len2 = len(str1), len(str2)
            if len1 > len2:
                return False
            if str2[:len1] != str1:
                return False
            return str2[-len1:] == str1

        count = 0
        for i in range(len(words)):
            for j in range(i + 1, len(words)):
                if isPrefixAndSuffix(words[i], words[j]):
                    count += 1
        return count