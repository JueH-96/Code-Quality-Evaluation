class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        n1, n2, n3 = len(s1), len(s2), len(s3)
        
        if not s1 or not s2 or not s3:
            return -1

        if s1[0] != s2[0] or s1[0] != s3[0]:
            return -1

        ans = 0
        i = n1 - 1
        j = n2 - 1
        k = n3 - 1

        while i >= 0 and j >= 0 and k >= 0:
            if s1[i] == s2[j] == s3[k]:
                i -= 1
                j -= 1
                k -= 1
            else:
                ans += 1
                if i > j and i > k:
                    i -= 1
                elif j > i and j > k:
                    j -= 1
                elif k > i and k > j:
                    k -= 1
                else:
                    return -1

        ans += max(0, i + 1)
        ans += max(0, j + 1)
        ans += max(0, k + 1)

        return ans