class Solution:
    def findMinimumOperations(self, s1: str, s2: str, s3: str) -> int:
        def common_suffix_length(a, b, c):
            min_len = min(len(a), len(b), len(c))
            for i in range(min_len):
                if a[-(i+1)] != b[-(i+1)] or a[-(i+1)] != c[-(i+1)]:
                    return i
            return min_len

        common_len = common_suffix_length(s1, s2, s3)
        if common_len == 0:
            return -1
        
        return (len(s1) - common_len) + (len(s2) - common_len) + (len(s3) - common_len)

# Example usage:
# sol = Solution()
# print(sol.findMinimumOperations("abc", "abb", "ab"))  # Output: 2
# print(sol.findMinimumOperations("dac", "bac", "cac"))  # Output: -1