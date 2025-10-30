class Solution:
    def countKConstraintSubstrings(self, s: str, k: int) -> int:
        n = len(s)
        # Create prefix sums for counts of '0' and '1'
        prefix0 = [0] * (n + 1)
        prefix1 = [0] * (n + 1)
        for i in range(n):
            prefix0[i + 1] = prefix0[i] + (1 if s[i] == '0' else 0)
            prefix1[i + 1] = prefix1[i] + (1 if s[i] == '1' else 0)
        
        count = 0
        # Enumerate all substrings using two nested loops.
        for i in range(n):
            for j in range(i, n):
                zeros = prefix0[j + 1] - prefix0[i]
                ones = prefix1[j + 1] - prefix1[i]
                # The substring satisfies the k-constraint if it has at most k zeros or at most k ones.
                if zeros <= k or ones <= k:
                    count += 1
        
        return count

# The following is for local testing,
# Feel free to use these test cases or modify them.

if __name__ == "__main__":
    sol = Solution()
    
    # Example 1
    s1 = "10101"
    k1 = 1
    print(sol.countKConstraintSubstrings(s1, k1))  # Expected output: 12

    # Example 2
    s2 = "1010101"
    k2 = 2
    print(sol.countKConstraintSubstrings(s2, k2))  # Expected output: 25

    # Example 3
    s3 = "11111"
    k3 = 1
    print(sol.countKConstraintSubstrings(s3, k3))  # Expected output: 15