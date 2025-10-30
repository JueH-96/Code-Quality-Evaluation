# YOUR CODE HERE
# This program solves the problem by reading 12 strings from standard input
# and counting how many of them have a length equal to their 1-based line number.

def solve():
    """
    Reads 12 strings and counts how many strings S_i have a length equal to i.
    """
    # Initialize a counter to store the number of strings that satisfy the condition.
    count = 0

    # The problem specifies 12 strings, S_1, S_2, ..., S_12.
    # A for loop iterating from 1 to 12 (inclusive) is a natural way to handle this.
    # The loop variable 'i' will represent the 1-based index of the string.
    for i in range(1, 13):
        # In each iteration, read the next string from standard input.
        # In the first iteration (i=1), this reads S_1.
        # In the second iteration (i=2), this reads S_2, and so on.
        s_i = input()
        
        # Check if the length of the string S_i is equal to its index i.
        if len(s_i) == i:
            # If the condition is true, increment the counter.
            count += 1

    # After the loop has processed all 12 strings,
    # print the final count to standard output.
    print(count)

# Execute the solution function.
solve()