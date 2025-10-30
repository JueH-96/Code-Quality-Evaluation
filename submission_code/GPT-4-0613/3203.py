from typing import List

class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        prefix = [[0]*26]
        for i in range(len(s)):
            count = prefix[-1][:]
            count[ord(s[i]) - ord('a')] += 1
            prefix.append(count)
        
        res = []
        for a, b, c, d in queries:
            count1 = [x - y for x, y in zip(prefix[b+1], prefix[a])]
            count2 = [x - y for x, y in zip(prefix[d+1], prefix[c])]
            res.append(sum(x % 2 for x in count1 + count2) // 2 <= d - c + 1)
        return res