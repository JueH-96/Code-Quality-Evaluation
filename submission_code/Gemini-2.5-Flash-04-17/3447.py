class Solution:
    def clearDigits(self, s: str) -> str:
        # Use a list to simulate a stack.
        # We iterate through the input string character by character.
        # If a character is a non-digit, we append it to our stack.
        # If a character is a digit, according to the rule "remove the closest non-digit
        # character to its left", the non-digit to remove is the last one we added
        # to our stack. So, we pop from the stack. The digit itself is not added.

        stack = []

        for char in s:
            if char.isdigit():
                # If the stack is not empty, pop the last non-digit added.
                # The problem guarantees that the input is such that it's possible
                # to delete all digits this way. This implies that when a digit
                # needs to be removed, there will be a non-digit to its left
                # that hasn't been removed yet. In our stack simulation, this means
                # the stack will not be empty when we encounter a digit that
                # corresponds to a removal step according to the rule.
                if stack: # Defensive check, although potentially guaranteed by problem constraints
                    stack.pop()
                # The digit itself is effectively removed by not adding it to the stack.
            else:
                # If it's a non-digit, push it onto the stack.
                stack.append(char)

        # The remaining characters in the stack form the resulting string.
        return "".join(stack)