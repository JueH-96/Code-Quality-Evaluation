class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        final_length = 0
        for char in s:
            steps_to_z = ord('z') - ord(char)
            if t <= steps_to_z:
                final_length = (final_length + 1) % MOD
            else:
                rem_t = t - steps_to_z
                final_length = (final_length + 1 + rem_t) % MOD
        return final_length