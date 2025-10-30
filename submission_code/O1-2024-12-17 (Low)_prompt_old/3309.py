class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        count = 0
        # Helper function to check if s1 is prefix and suffix of s2
        def isPrefixAndSuffix(s1, s2):
            return s2.startswith(s1) and s2.endswith(s1)
        
        # Iterate over all valid pairs (i, j) with i < j
        for i in range(len(words) - 1):
            for j in range(i + 1, len(words)):
                if isPrefixAndSuffix(words[i], words[j]):
                    count += 1
        
        return count