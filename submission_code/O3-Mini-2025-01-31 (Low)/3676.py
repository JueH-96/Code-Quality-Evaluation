class Solution:
    def smallestNumber(self, n: int) -> int:
        # Determine the bit length of n
        bit_length = n.bit_length()
        # Candidate number with all bits set
        candidate = (1 << bit_length) - 1
        
        # If candidate is less than n, use next bit length
        if candidate < n:
            candidate = (1 << (bit_length + 1)) - 1
        
        return candidate

# You can test the function using the following lines:
if __name__ == "__main__":
    sol = Solution()
    test_cases = [5, 10, 3]
    for tc in test_cases:
        print("Input:", tc, "Output:", sol.smallestNumber(tc))