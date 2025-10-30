class Solution:
    def countPrefixSuffixPairs(self, words: List[str]) -> int:
        
        def isPrefixAndSuffix(str1, str2):
            # Find the common prefix and suffix lengths simultaneously.
            m, n = len(str1), len(str2)
            prefix_len, suffix_len = 0, 0
            # Prefix scan
            for i in range(min(m, n)):
                if str1[i] != str2[i]:
                    break
                prefix_len += 1
            # Suffix scan
            for i in range(min(m, n) - 1, -1, -1):
                if str1[m - 1 - (i - prefix_len)] != str2[n - 1 - (i - prefix_len)]:
                    break
                suffix_len += 1
            
            return (prefix_len + suffix_len) >= m
        
        ans = 0
        for i in range(len(words) - 1):
            for j in range(i + 1, len(words)):
                if isPrefixAndSuffix(words[i], words[j]):
                    ans += 1
        return ans