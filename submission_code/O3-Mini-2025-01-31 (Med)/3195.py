class Solution:
    def minimumSteps(self, s: str) -> int:
        ones = 0   # Count of black balls seen so far (represented by '1')
        steps = 0  # Total number of swaps required

        # For each ball in the string, if it's black, increment ones.
        # If it's white ('0'), add the number of black balls seen before it.
        # This counts the inversions (i.e., each black ball that is to the left of a white ball)
        # which is equal to the number of adjacent swaps needed.
        for char in s:
            if char == '1':
                ones += 1
            else:
                steps += ones

        return steps

# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.minimumSteps("101"))   # Output: 1
    print(sol.minimumSteps("100"))   # Output: 2
    print(sol.minimumSteps("0111"))  # Output: 0