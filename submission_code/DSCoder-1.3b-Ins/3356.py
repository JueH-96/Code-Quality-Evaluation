from typing import List

class Solution:
    def shortestSubstrings(self, arr: List[str]) -> List[str]:
        n = len(arr)
        prefix = [[0]*26 for _ in range(n)]
        suffix = [[0]*26 for _ in range(n)]
        for i in range(n):
            for j in range(26):
                prefix[i][j] = arr[i][:i+1].count(chr(j+97)) if i else j
                suffix[i][j] = arr[i][i-j:].count(chr(j+97)) if i-j >= 0 else j
        res = [""]*n
        for i in range(n):
            for j in range(26):
                k = prefix[i][j] + suffix[i][j] - 1
                if k < len(res[i]):
                    res[i] = arr[i][k]
        return res