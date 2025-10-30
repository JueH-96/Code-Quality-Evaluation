class Solution:
    def longestCommonPrefix(self, arr1: List[int], arr2: List[int]) -> int:
        def get_prefix_len(s1, s2):
            n = min(len(s1), len(s2))
            count = 0
            for i in range(n):
                if s1[i] == s2[i]:
                    count += 1
                else:
                    break
            return count

        ans = 0
        for x in arr1:
            for y in arr2:
                ans = max(ans, get_prefix_len(str(x), str(y)))
        return ans