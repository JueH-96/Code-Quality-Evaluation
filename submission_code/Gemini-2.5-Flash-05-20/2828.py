class Solution:
    def smallestString(self, s: str) -> str:
        s_list = list(s)
        n = len(s)

        # 1. Find the first character that is not 'a'
        # This will be the start of our substring for the operation.
        start_idx = -1
        for i in range(n):
            if s_list[i] != 'a':
                start_idx = i
                break

        # 2. Handle the special case where all characters in s are 'a'
        if start_idx == -1:
            # If the string consists only of 'a's (e.g., "aaa"),
            # any operation will turn at least one 'a' into 'z'.
            # To get the lexicographically smallest string, we want the 'z'
            # to appear as late as possible. So, we change only the last 'a'.
            # Example: "aaa" -> "aaz"
            s_list[n - 1] = 'z'
        else:
            # 3. If there's at least one non-'a' character:
            # The operation must start at `start_idx` (the first non-'a' character).
            # We want to continue applying the operation to subsequent characters
            # as long as they are not 'a' (because 'b' -> 'a' is beneficial).
            # We must stop just before the first 'a' we encounter after `start_idx`,
            # because 'a' -> 'z' is detrimental and would make the string larger.
            
            end_idx = start_idx # Initialize end_idx to start_idx
            # Iterate from start_idx to find where the operation should end.
            for i in range(start_idx, n):
                if s_list[i] == 'a':
                    # We hit an 'a'. This 'a' should not be part of the operation,
                    # as it would become 'z'. So, the operation ends at the
                    # character immediately before this 'a'.
                    end_idx = i - 1 
                    break
                else:
                    # This character is not 'a', so we include it in the operation.
                    # Update end_idx to the current character's index.
                    end_idx = i 
            
            # 4. Apply the operation to the identified substring
            # All characters in the range [start_idx, end_idx] are guaranteed to be non-'a'.
            for i in range(start_idx, end_idx + 1):
                s_list[i] = chr(ord(s_list[i]) - 1)
        
        # 5. Join the list of characters back into a string
        return "".join(s_list)