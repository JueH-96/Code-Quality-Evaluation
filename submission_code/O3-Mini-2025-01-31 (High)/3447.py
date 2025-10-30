class Solution:
    def clearDigits(self, s: str) -> str:
        # We can simulate the deletion process by using a stack.
        # For every character in s:
        #   - If it's a letter, push it onto the stack.
        #   - If it's a digit, then "delete" it by not adding it,
        #     and also remove (pop) the last letter from the stack.
        # This is equivalent to repeatedly deleting the first digit (in the
        # remaining string) and the closest non-digit to its left.
        stack = []
        for ch in s:
            if ch.isdigit():
                # Delete the digit and also remove the letter immediately to its left.
                # The stack's top holds the last letter that has not been removed.
                if stack:  # Guaranteed by the problem's constraints.
                    stack.pop()
            else:
                stack.append(ch)
        return ''.join(stack)