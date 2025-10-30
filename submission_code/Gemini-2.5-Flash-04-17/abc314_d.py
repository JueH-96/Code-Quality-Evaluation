# YOUR CODE HERE
import sys

# The problem involves potentially large inputs (N, Q up to 5*10^5),
# so using sys.stdin.readline for efficient input reading is necessary.
# Standard input/output is required.

def solve():
    # Read N, the length of the string
    N = int(sys.stdin.readline())

    # Read the initial string S and convert it to a mutable list of characters
    # Lists are used because strings are immutable in Python, and we need to modify characters
    S = list(sys.stdin.readline().strip())

    # Read Q, the number of operations
    Q = int(sys.stdin.readline())

    # Read all Q operations and store them
    # Each operation is a tuple (t_i, x_i, c_i)
    ops = []
    for _ in range(Q):
        # Read the line and split it into components
        parts = sys.stdin.readline().split()
        t = int(parts[0]) # Operation type
        x = int(parts[1]) # Index (1-based) for type 1, 0 for type 2/3
        c = parts[2]      # Character for type 1, 'a' for type 2/3 (ignored)
        ops.append((t, x, c))

    # The core optimization: The global case conversion effect is only determined
    # by the *last* Type 2 or Type 3 operation. Operations of type 2 or 3 before
    # the last one only have temporary effects that are overwritten.
    # We find the index of this last global case conversion operation.

    last_global_op_idx = -1 # Initialize index to -1 (means no global op found yet)
    last_global_op_type = None # Store the type (2 or 3)

    # Iterate backwards through the operations to find the last one of type 2 or 3
    for i in range(Q - 1, -1, -1):
        if ops[i][0] in (2, 3):
            # Found the last global case conversion operation
            last_global_op_idx = i
            last_global_op_type = ops[i][0]
            break # Exit the loop once the last one is found

    # Initialize the character list that will be modified.
    # We reuse the list read from stdin to avoid extra memory allocation.
    char_list = S

    # Process all operations in chronological order (from first to last)
    for i in range(Q):
        op_type, x, c = ops[i]

        if i < last_global_op_idx:
            # If the current operation is before the last global case conversion:
            # We only care about Type 1 operations here.
            # Type 2/3 operations before the last one have their global effect
            # overridden by the last global operation.
            if op_type == 1:
                # Apply the character change directly to the list
                # x is 1-based index, so list index is x - 1
                char_list[x - 1] = c
            # Type 2/3 operations (op_type == 2 or 3) are ignored in this block.

        elif i == last_global_op_idx:
            # This is the specific operation that performs the final global case conversion.
            # Apply the determined last global conversion type to the entire current string state.
            # This step takes O(N) time.
            if last_global_op_type == 2: # Convert all characters to lowercase
                for j in range(N):
                    char_list[j] = char_list[j].lower()
            elif last_global_op_type == 3: # Convert all characters to uppercase
                 for j in range(N):
                    char_list[j] = char_list[j].upper()
            # Note: By definition of last_global_op_idx, ops[i][0] is 2 or 3 here.
            # Problem constraints state if t_i != 1, then x_i=0 and c_i='a',
            # so Type 1 ops cannot occur at the index of a global op.

        else: # i > last_global_op_idx
            # If the current operation is after the last global case conversion:
            # Only Type 1 operations can occur here (by definition of last_global_op_idx).
            # Apply the character change directly. This overrides any previous global case.
            if op_type == 1:
                 # Apply the character change directly to the list
                 # x is 1-based index, so list index is x - 1
                 char_list[x - 1] = c
            # Type 2/3 operations cannot occur here.

    # After processing all operations, the char_list contains the final state of the string.
    # Join the list of characters back into a single string.
    result_string = "".join(char_list)

    # Print the final string
    print(result_string)

# Call the solve function to run the program
solve()