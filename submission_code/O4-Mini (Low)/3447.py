class Solution:
    def clearDigits(self, s: str) -> str:
        stack = []
        for ch in s:
            if ch.isdigit():
                # Remove the closest non-digit character to the left
                # (which must be on top of our stack)
                if stack:
                    stack.pop()
                # We skip adding this digit
            else:
                # Non-digit characters get pushed onto the stack
                stack.append(ch)
        # The stack now contains the resulting characters
        return "".join(stack)


# Example usage:
if __name__ == "__main__":
    sol = Solution()
    print(sol.clearDigits("abc"))   # Expected "abc"
    print(sol.clearDigits("cb34"))  # Expected ""
    print(sol.clearDigits("a1b2c3"))# Expected ""
    print(sol.clearDigits("a2b3c")) # Expected "c"