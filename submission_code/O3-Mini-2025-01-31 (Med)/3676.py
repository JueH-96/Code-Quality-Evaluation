class Solution:
    def smallestNumber(self, n: int) -> int:
        k = n.bit_length()  # The number of bits required to represent n
        candidate = (1 << k) - 1  # This produces a number with k set bits (all ones)
        
        # If the candidate is less than n, we need to use one more bit
        if candidate < n:
            candidate = (1 << (k + 1)) - 1
        return candidate

# Example test cases
if __name__ == "__main__":
    sol = Solution()
    print(sol.smallestNumber(5))   # Expected output: 7
    print(sol.smallestNumber(10))  # Expected output: 15
    print(sol.smallestNumber(3))   # Expected output: 3