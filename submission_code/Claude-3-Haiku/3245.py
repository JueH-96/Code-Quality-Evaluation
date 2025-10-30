class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        result = []
        for i in range(len(s) - len(a) + 1):
            if s[i:i+len(a)] == a:
                for j in range(len(s) - len(b) + 1):
                    if s[j:j+len(b)] == b and abs(j - i) <= k:
                        result.append(i)
                        break
        return result