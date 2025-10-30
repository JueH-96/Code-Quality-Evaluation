class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        n = len(s)
        len_a = len(a)
        len_b = len(b)

        indices_a = []
        for i in range(n - len_a + 1):
            if s[i:i + len_a] == a:
                indices_a.append(i)

        indices_b = []
        for i in range(n - len_b + 1):
            if s[i:i + len_b] == b:
                indices_b.append(i)

        beautiful_indices_set = set()
        for i in indices_a:
            for j in indices_b:
                if abs(j - i) <= k:
                    beautiful_indices_set.add(i)
                    break

        beautiful_indices = sorted(list(beautiful_indices_set))
        return beautiful_indices