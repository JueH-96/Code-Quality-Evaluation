class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        def find_occurrences(sub: str) -> List[int]:
            return [i for i in range(len(s) - len(sub) + 1) if s[i:i + len(sub)] == sub]

        a_indices = find_occurrences(a)
        b_indices = find_occurrences(b)

        beautiful_indices = []

        j = 0
        for i in a_indices:
            while j < len(b_indices) and b_indices[j] < i - k:
                j += 1
            if j < len(b_indices) and abs(b_indices[j] - i) <= k:
                beautiful_indices.append(i)
            elif j > 0 and abs(b_indices[j - 1] - i) <= k:
                beautiful_indices.append(i)

        return beautiful_indices