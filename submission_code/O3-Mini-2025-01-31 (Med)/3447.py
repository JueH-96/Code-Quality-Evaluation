class Solution:
    def clearDigits(self, s: str) -> str:
        # We'll use a stack to simulate the removals.
        # For every character in s:
        # If it's a letter, we push it onto the stack.
        # If it's a digit, we remove (pop) the last letter from the stack.
        stack = []
        for ch in s:
            if not ch.isdigit():
                stack.append(ch)
            else:
                # Since the problem guarantees that deletion of all digits is possible,
                # we know there is always a non-digit character on the left.
                if stack:
                    stack.pop()
        # The remaining characters in the stack form the processed string.
        return ''.join(stack)

# Example usage:
# sol = Solution()
# print(sol.clearDigits("cb34"))  # Output: ""
# print(sol.clearDigits("abc"))  # Output: "abc"