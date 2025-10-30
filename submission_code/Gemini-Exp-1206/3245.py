class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        n = len(s)
        a_indices = []
        b_indices = []
        for i in range(n):
            if i + len(a) <= n and s[i:i + len(a)] == a:
                a_indices.append(i)
            if i + len(b) <= n and s[i:i + len(b)] == b:
                b_indices.append(i)
        result = []
        for i in a_indices:
            for j in b_indices:
                if abs(i - j) <= k:
                    result.append(i)
                    break
        result.sort()
        return result