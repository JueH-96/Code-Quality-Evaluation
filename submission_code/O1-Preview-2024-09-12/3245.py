class Solution:
    def beautifulIndices(self, s: str, a: str, b: str, k: int) -> List[int]:
        from bisect import bisect_left, bisect_right
        len_a = len(a)
        len_b = len(b)
        n = len(s)
        indices_a = []
        indices_b = []
        for i in range(n - len_a +1):
            if s[i:i+len_a] == a:
                indices_a.append(i)
        for i in range(n - len_b +1):
            if s[i:i+len_b] == b:
                indices_b.append(i)
        indices_b.sort()
        result = []
        for i in indices_a:
            left = bisect_left(indices_b, i - k)
            right = bisect_right(indices_b, i + k)
            if left < right:
                result.append(i)
        return result