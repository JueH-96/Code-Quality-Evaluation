class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        total = 0
        # Iterate over each possible starting index of substring
        for i in range(n):
            count0, count1 = 0, 0
            # Extend the substring from index i to the end
            for j in range(i, n):
                if s[j] == '0':
                    count0 += 1
                else:
                    count1 += 1
                # If either the count of '0's or the count of '1's is <= k,
                # then this substring satisfies the k-constraint.
                if count0 <= k or count1 <= k:
                    total += 1
        return total

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    
    # Example 1:
    s = "10101"
    k = 1
    print(sol.countKConstraintSubstrings(s, k))  # Expected output: 12
    
    # Example 2:
    s = "1010101"
    k = 2
    print(sol.countKConstraintSubstrings(s, k))  # Expected output: 25
    
    # Example 3:
    s = "11111"
    k = 1
    print(sol.countKConstraintSubstrings(s, k))  # Expected output: 15