class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        a_indices = []
        b_indices = []
        n = len(s)
        len_a = len(a)
        len_b = len(b)

        for i in range(n - len_a + 1):
            if s[i:i + len_a] == a:
                a_indices.append(i)

        for j in range(n - len_b + 1):
            if s[j:j + len_b] == b:
                b_indices.append(j)

        beautiful_indices = []
        for i in a_indices:
            is_beautiful = False
            for j in b_indices:
                if abs(j - i) <= k:
                    is_beautiful = True
                    break
            if is_beautiful:
                beautiful_indices.append(i)

        return beautiful_indices