class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        n = len(s)
        z_count = s.count('z')

        # The length after t transformations
        length_after_t = n

        # For each transformation, the length increases by the number of 'z' characters
        for _ in range(t):
            length_after_t += z_count
            z_count *= 2
            z_count %= MOD
            length_after_t %= MOD

        return length_after_t