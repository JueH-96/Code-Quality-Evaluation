class Solution:
    def smallestNumber(self, n: int) -> int:
        # Start from i = 1 bits and increase
        i = 1
        while True:
            # Create a number with i set bits: (2 ** i) - 1 has binary representation of i ones.
            candidate = (1 << i) - 1
            if candidate >= n:
                return candidate
            i += 1

# Example test cases
if __name__ == "__main__":
    sol = Solution()
    print(sol.smallestNumber(5))  # Output: 7 (binary "111")
    print(sol.smallestNumber(10)) # Output: 15 (binary "1111")
    print(sol.smallestNumber(3))  # Output: 3 (binary "11")