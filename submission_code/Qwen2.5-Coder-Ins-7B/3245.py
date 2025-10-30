class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        indices_a = [i for i in range(len(s) - len(a) + 1) if s[i:i + len(a)] == a]
        indices_b = [j for j in range(len(s) - len(b) + 1) if s[j:j + len(b)] == b]
        
        beautiful_indices = []
        for i in indices_a:
            for j in indices_b:
                if abs(i - j) <= k:
                    beautiful_indices.append(i)
                    break
        
        return sorted(set(beautiful_indices))