class Solution:
    def lengthAfterTransformations(self, s: str, t: int) -> int:
        MOD = 10**9 + 7
        
        # Initial length of the string
        length = len(s)
        
        # Each 'z' will be replaced by "ab" which increases the length by 1
        # Each other character will just be replaced by the next character, keeping the length same
        # So we need to count the number of 'z' in the initial string
        z_count = s.count('z')
        
        # After each transformation, the number of 'z' will increase by the number of 'z' in the previous step
        # So we need to calculate the total length after t transformations
        for _ in range(t):
            length = (length + z_count) % MOD
            z_count = (z_count * 2) % MOD
        
        return length

# Example usage:
# sol = Solution()
# print(sol.lengthAfterTransformations("abcyy", 2))  # Output: 7
# print(sol.lengthAfterTransformations("azbk", 1))   # Output: 5