class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        n = len(s)
        la = len(a)
        lb = len(b)
        ans = []
        b_indices = []
        for i in range(n - lb + 1):
            if s[i:i + lb] == b:
                b_indices.append(i)

        for i in range(n - la + 1):
            if s[i:i + la] == a:
                for j in b_indices:
                    if abs(i - j) <= k:
                        ans.append(i)
                        break
        return ans