class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> list[int]:
        a_indices = [i for i in range(len(s)) if s[i:i+len(a)] == a]
        b_indices = [i for i in range(len(s)) if s[i:i+len(b)] == b]
        beautiful_indices = []
        for a_index in a_indices:
            for b_index in b_indices:
                if abs(a_index - b_index) <= k:
                    beautiful_indices.append(a_index)
                    break
        return sorted(beautiful_indices)