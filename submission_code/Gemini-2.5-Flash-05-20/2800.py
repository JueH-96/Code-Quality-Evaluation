class Solution:
    def minLength(self, s: str) -> int:
        """
        Calculates the minimum possible length of a string by repeatedly
        removing occurrences of "AB" or "CD".

        The approach uses a stack to process the string. When a character
        is encountered, it's pushed onto the stack. If the last two
        characters on the stack form "AB" or "CD", they are popped,
        effectively removing them from the string and allowing previously
        non-adjacent characters to become adjacent and potentially form
        new removable pairs. This greedy strategy ensures the minimum length
        because the order of removals of non-overlapping patterns doesn't matter,
        and for overlapping patterns, the stack naturally resolves them by
        processing from left to right and handling newly formed adjacencies.

        Args:
            s: The input string consisting only of uppercase English letters.

        Returns:
            The minimum possible length of the resulting string.
        """
        
        stack = []
        
        for char in s:
            stack.append(char)
            
            # Check if the last two characters form "AB" or "CD"
            # This check is only relevant if the stack has at least two elements
            if len(stack) >= 2:
                top_char = stack[-1]
                second_top_char = stack[-2]
                
                # Condition for removal: "AB" or "CD"
                if (second_top_char == 'A' and top_char == 'B') or \
                   (second_top_char == 'C' and top_char == 'D'):
                    stack.pop() # Remove the top character ('B' or 'D')
                    stack.pop() # Remove the second top character ('A' or 'C')
                    
        # The length of the remaining stack is the minimum possible length
        return len(stack)