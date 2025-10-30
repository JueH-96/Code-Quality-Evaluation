class Solution:
    def colorTheArray(self, n: int, queries: List[List[int]]) -> List[int]:
        colors = [0] * n
        result = 0
        ans = []
        for i, c in queries:
            if colors[i] == c:
                ans.append(result)
                continue
            prev_color = colors[i]
            colors[i] = c
            delta = 0
            for dj in (-1, 1):
                j = i + dj
                if 0 <= j < n:
                    old_color = colors[j]
                    # Old contribution
                    if prev_color == old_color and prev_color != 0:
                        old_contrib = 1
                    else:
                        old_contrib = 0
                    # New contribution
                    if c == old_color and c != 0:
                        new_contrib = 1
                    else:
                        new_contrib = 0
                    delta += (new_contrib - old_contrib)
            result += delta
            ans.append(result)
        return ans