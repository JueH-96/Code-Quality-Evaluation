class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        def isPrefixAndSuffix(str1, str2):
            # str1 must be both a prefix and suffix of str2
            # First check if str1 is longer than str2 (impossible to be prefix/suffix)
            if len(str1) > len(str2):
                return False
            
            # Check if str1 is a prefix of str2
            if not str2.startswith(str1):
                return False
            
            # Check if str1 is a suffix of str2
            if not str2.endswith(str1):
                return False
            
            return True
        
        count = 0
        n = len(words)
        
        # Check all pairs (i, j) where i < j
        for i in range(n):
            for j in range(i + 1, n):
                if isPrefixAndSuffix(words[i], words[j]):
                    count += 1
        
        return count