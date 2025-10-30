class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        max_len = 0
        for num1 in arr1:
            for num2 in arr2:
                s1 = str(num1)
                s2 = str(num2)
                lcp_len = 0
                for i in range(min(len(s1), len(s2))):
                    if s1[i] == s2[i]:
                        lcp_len += 1
                    else:
                        break
                max_len = max(max_len, lcp_len)
        return max_len