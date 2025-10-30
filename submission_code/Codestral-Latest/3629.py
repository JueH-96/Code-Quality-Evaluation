class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7

        def transform_char(c):
            if c == 'z':
                return "ab"
            else:
                return chr((ord(c) - ord('a') + 1) % 26 + ord('a'))

        def transform_string(s):
            new_s = []
            for c in s:
                new_s.append(transform_char(c))
            return ''.join(new_s)

        length = len(s)
        for _ in range(t):
            s = transform_string(s)
            length = len(s)

        return length % MOD