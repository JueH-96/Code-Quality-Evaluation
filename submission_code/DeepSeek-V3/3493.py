class Solution:
    def maxOperations(self, s: str) -> int:
        n = len(s)
        operations = 0
        # Convert the string to a list for easier manipulation
        s_list = list(s)
        # Iterate through the string to find all '1's and count the number of '0's after them
        # We will use a list to keep track of the positions of '1's
        ones = []
        for i in range(n):
            if s_list[i] == '1':
                ones.append(i)
        # Now, for each '1', count the number of '0's to its right
        # The number of operations is the sum of the number of '0's after each '1'
        # But we need to ensure that the '1's are processed in order
        # So, we iterate through the '1's and for each, count the '0's after it
        # and subtract the '0's that have been used by previous '1's
        # We can use a pointer to keep track of the current position in the string
        # Initialize the pointer to the first '1'
        if not ones:
            return 0
        ptr = ones[0]
        for i in range(len(ones)):
            current_one = ones[i]
            # Count the number of '0's after current_one
            # but only up to the next '1' or the end of the string
            # and subtract the '0's that have been used by previous '1's
            # The number of '0's available is the number of '0's between current_one and the next '1'
            # or the end of the string
            if i < len(ones) - 1:
                next_one = ones[i+1]
            else:
                next_one = n
            # Count the '0's between current_one and next_one
            zeros = 0
            for j in range(current_one + 1, next_one):
                if s_list[j] == '0':
                    zeros += 1
            operations += zeros
        return operations