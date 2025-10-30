class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        # Build prefix sums for zeros and ones.
        zeros = [0] * (n + 1)
        ones = [0] * (n + 1)
        for i in range(n):
            zeros[i+1] = zeros[i] + (1 if s[i] == '0' else 0)
            ones[i+1] = ones[i] + (1 if s[i] == '1' else 0)
        
        count = 0
        # Iterate over all substrings [i, j]
        for i in range(n):
            for j in range(i, n):
                count0 = zeros[j+1] - zeros[i]
                count1 = ones[j+1] - ones[i]
                # Substring meets k constraint if either zeros <= k or ones <= k.
                if count0 <= k or count1 <= k:
                    count += 1
        return count

# For quick local testing.
if __name__ == '__main__':
    sol = Solution()
    print(sol.countKConstraintSubstrings("10101", 1))  # Expected 12
    print(sol.countKConstraintSubstrings("1010101", 2))  # Expected 25
    print(sol.countKConstraintSubstrings("11111", 1))  # Expected 15