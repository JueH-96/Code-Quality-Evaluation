class Solution:
    def smallestString(self, s: str) -> str:
        n = len(s)
        chars = list(s)

        # Find the index of the first character that is not 'a'.
        # This index will be the optimal starting point for the operation
        # because changing an 'a' to 'z' increases its value, which is
        # detrimental lexicographically at an early position.
        first_non_a = -1
        for i in range(n):
            if chars[i] != 'a':
                first_non_a = i
                break

        # If all characters are 'a', the only way to change the string
        # is to turn some 'a's into 'z's. To get the lexicographically
        # smallest result, we should introduce 'z' as late as possible.
        # This is achieved by changing only the last 'a' to 'z'.
        if first_non_a == -1:
            # The problem requires performing the operation exactly once on a non-empty substring.
            # If the string is all 'a's, changing the last character 'a' to 'z'
            # creates the string 'a...az' which is the smallest possible
            # result amongst strings containing at least one 'z' (all other options
            # would put a 'z' at an earlier position or add more 'z's).
            chars[n - 1] = 'z'
            return "".join(chars)

        # If there is at least one non-'a' character, the operation must
        # start at the first non-'a' character (index first_non_a) to get
        # the largest lexicographical decrease at the earliest position.
        start_op = first_non_a

        # The operation should continue on subsequent characters as long as they are not 'a'.
        # If we encounter an 'a', applying the operation would turn it into 'z', which
        # would make the string lexicographically larger at that position compared to leaving it as 'a'.
        # So, we stop the operation just before the first 'a' encountered at or after start_op.
        # Find the index of the first 'a' at or after start_op.
        end_op = n # Default to the end of the string if no 'a' is found
        for j in range(start_op, n):
            if chars[j] == 'a':
                end_op = j
                break

        # Apply the operation (decrement character, e.g., 'b' -> 'a', 'c' -> 'b')
        # to the substring from start_op up to (but not including) end_op.
        # The characters in this range (chars[k] for start_op <= k < end_op)
        # are guaranteed not to be 'a', based on how start_op and end_op were determined.
        # The transformation `chr(ord(char) - 1)` is correct for characters 'b' through 'z'.
        for k in range(start_op, end_op):
            chars[k] = chr(ord(chars[k]) - 1)

        # Join the modified list of characters back into a string.
        return "".join(chars)