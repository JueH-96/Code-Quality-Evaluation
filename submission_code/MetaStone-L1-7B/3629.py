class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        mod = 10**9 + 7
        total = 0
        for c in s:
            if c == 'z':
                if t == 0:
                    total += 1
                else:
                    total += 2
            else:
                steps_to_z = 25 - (ord(c) - ord('a'))
                if t <= steps_to_z:
                    total += 1
                else:
                    total += 2
        return total % mod