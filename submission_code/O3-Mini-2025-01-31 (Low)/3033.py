class Solution:
    def minOperations(self, s1: str, s2: str, x: int) -> int:
        n = len(s1)
        # record positions where s1 and s2 differ
        diff = []
        for i in range(n):
            if s1[i] != s2[i]:
                diff.append(i)
        
        # If the count of differences is odd, we cannot fix them with operations flipping two bits.
        if len(diff) % 2 == 1:
            return -1
        
        # When exactly two differences exist, we may “slide” the one difference to meet the other,
        # or use the direct arbitrary flip.
        if len(diff) == 2:
            # Option 1: use adjacent-flip(s) to “move” the difference: cost = distance between indices.
            cost_adj = diff[1] - diff[0]
            # Option 2: use one arbitrary flip which costs x.
            return min(x, cost_adj)
        
        # For all other cases (including zero differences), the optimal strategy is to fix differences
        # in pairs using the arbitrary flip operation.
        # (One may prove that if there are more than 2 differences, “sliding” them does not reduce cost.)
        return (len(diff) // 2) * x

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    # Example 1:
    print(sol.minOperations("1100011000", "0101001010", 2))  # Expected output: 4
    # Example 2:
    print(sol.minOperations("10110", "00011", 4))            # Expected output: -1