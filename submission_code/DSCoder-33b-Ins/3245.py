class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        indices_a = [i for i in range(len(s)) if s[i:i+len(a)] == a]
        indices_b = [i for i in range(len(s)) if s[i:i+len(b)] == b]
        beautiful_indices = []
        for i in indices_a:
            for j in indices_b:
                if abs(i - j) <= k:
                    beautiful_indices.append(i)
                    break
        return sorted(beautiful_indices)