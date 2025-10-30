class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        original_length = len(s)
        threshold = 25 - t
        count = 0
        for c in s:
            c_val = ord(c) - ord('a')
            if c_val > threshold:
                count += 1
        return (original_length + count) % MOD