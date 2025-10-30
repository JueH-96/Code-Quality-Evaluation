class Solution:
    def canMakePalindromeQueries(self, s: str, queries: List[List[int]]) -> List[bool]:
        n = len(s)
        prefix_counts = [[0] * 26]
        for i in range(n):
            counts = prefix_counts[-1][:]
            counts[ord(s[i]) - ord('a')] += 1
            prefix_counts.append(counts)

        ans = []
        for a, b, c, d in queries:
            diff_counts = [0] * 26
            for i in range(26):
                diff_counts[i] = prefix_counts[b + 1][i] - prefix_counts[a][i] - (prefix_counts[d + 1][i] - prefix_counts[c][i])
            
            odd_count = 0
            for count in diff_counts:
                if count % 2 != 0:
                    odd_count += 1
            
            ans.append(odd_count <= 1)
        return ans