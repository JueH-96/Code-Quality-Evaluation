class Solution:
    def smallestNumber(self, n: int) -> int:
        # We want the smallest number x = 2^k - 1 such that x >= n.
        # Compute the bit length of n
        k = n.bit_length()
        candidate = (1 << k) - 1
        if candidate >= n:
            return candidate
        # Otherwise increase k by 1
        return (1 << (k + 1)) - 1


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.smallestNumber(5))   # Output: 7
    print(sol.smallestNumber(10))  # Output: 15
    print(sol.smallestNumber(3))   # Output: 3