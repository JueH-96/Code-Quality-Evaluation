class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        prefix_counts = [[0]*26]
        for i in range(n):
            prefix_counts.append(prefix_counts[-1][:])
            prefix_counts[-1][ord(s[i]) - ord('a')] += 1

        res = []
        for a, b, c, d in queries:
            count_a = prefix_counts[b+1]
            count_b = prefix_counts[d+1]
            count_c = [count_b[i] - count_a[i] for i in range(26)]
            odds = sum(x % 2 for x in count_c)
            res.append(odds // 2 <= b - a + 1)
        return res