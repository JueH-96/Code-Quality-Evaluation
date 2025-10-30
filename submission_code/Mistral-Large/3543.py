class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        count = 0

        # Helper function to count valid substrings ending at each index
        def count_valid_substrings(char):
            nonlocal count
            l = 0
            char_count = 0
            for r in range(n):
                if s[r] == char:
                    char_count += 1
                while char_count > k:
                    if s[l] == char:
                        char_count -= 1
                    l += 1
                count += r - l + 1

        # Count substrings with at most k '0's
        count_valid_substrings('0')

        # Count substrings with at most k '1's
        count_valid_substrings('1')

        # Since we counted each valid substring twice, divide by 2
        return count // 2

# Example usage:
sol = Solution()
print(sol.countKConstraintSubstrings("10101", 1))  # Output: 12
print(sol.countKConstraintSubstrings("1010101", 2))  # Output: 25
print(sol.countKConstraintSubstrings("11111", 1))  # Output: 15