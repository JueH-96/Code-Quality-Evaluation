class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        total_length = 0
        mod = 10**9 + 7
        for char in s:
            if char == 'z':
                total_length += 2
            else:
                diff = ord('z') - ord(char)
                if t > diff:
                    total_length += 2
                else:
                    total_length += 1
        return total_length % mod