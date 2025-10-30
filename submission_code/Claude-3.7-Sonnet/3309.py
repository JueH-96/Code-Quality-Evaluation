class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        def isPrefixAndSuffix(str1, str2):
            # If str1 is longer than str2, it can't be a prefix or suffix
            if len(str1) > len(str2):
                return False
            # Check if str1 is both a prefix and a suffix of str2
            return str2.startswith(str1) and str2.endswith(str1)
        
        count = 0
        # Check all pairs (i,j) where i < j
        for i in range(len(words)):
            for j in range(i+1, len(words)):
                if isPrefixAndSuffix(words[i], words[j]):
                    count += 1
                    
        return count