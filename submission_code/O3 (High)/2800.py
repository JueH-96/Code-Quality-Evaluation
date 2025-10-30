class Solution:
    def minLength(self, s: str) -> int:
        """
        Scan the string once, keeping a stack with the characters that are
        still present after all removals performed so far.
        
        Whenever the current character together with the character on the
        stack’s top forms one of the removable substrings (“AB” or “CD”),
        we delete that pair (i.e. pop the stack and skip pushing the current
        character).  
        Otherwise we simply push the current character.
        
        Because the second letter of every removable substring (‘B’ or ‘D’)
        is never the first letter of another removable substring, every pair
        that can possibly be deleted will be discovered the moment its second
        character is processed, so this greedy single-pass procedure always
        yields the shortest possible string.
        
        Time complexity : O(n)  
        Space complexity: O(n)   (at most all characters stay in the stack)
        """
        stack = []
        for ch in s:
            if stack and (
                (stack[-1] == 'A' and ch == 'B') or
                (stack[-1] == 'C' and ch == 'D')
            ):
                stack.pop()      # remove the pair
            else:
                stack.append(ch) # keep the character
        return len(stack)