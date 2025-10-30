class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        def isPrefixAndSuffix(str1, str2):
            # str1 must be shorter than or equal to str2 to be both prefix and suffix
            if len(str1) > len(str2):
                return False
            
            # Check if str1 is both prefix and suffix of str2
            return str2.startswith(str1) and str2.endswith(str1)
        
        count = 0
        n = len(words)
        
        # Check all pairs (i, j) where i < j
        for i in range(n):
            for j in range(i + 1, n):
                if isPrefixAndSuffix(words[i], words[j]):
                    count += 1
        
        return count