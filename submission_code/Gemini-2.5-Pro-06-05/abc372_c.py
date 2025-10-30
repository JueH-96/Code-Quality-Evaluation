import sys

def solve():
    """
    Solves the problem by maintaining a running count of "ABC" substrings
    and updating it efficiently for each query.
    """
    # Read problem size and initial string from standard input
    try:
        N_str, Q_str = sys.stdin.readline().split()
        N = int(N_str)
        Q = int(Q_str)
        S = sys.stdin.readline().strip()
    except (IOError, ValueError):
        # Gracefully handle empty input at the end of file
        return

    # Use a list of characters for efficient, mutable string operations
    s_list = list(S)

    def is_abc(i):
        """
        Helper function to check if the substring starting at index i is "ABC".
        It handles boundary conditions, returning False for invalid indices.
        """
        # The start index `i` must be within the valid range [0, N-3]
        if not (0 <= i <= N - 3):
            return False
        # Check if the characters form "ABC"
        return s_list[i] == 'A' and s_list[i+1] == 'B' and s_list[i+2] == 'C'

    # --- Initial Calculation ---
    # Calculate the initial number of "ABC" occurrences in O(N) time.
    count = 0
    for i in range(N - 2):
        if is_abc(i):
            count += 1

    # --- Process Queries ---
    # Process each of the Q queries in O(1) time.
    for _ in range(Q):
        # Read the query details
        line = sys.stdin.readline().split()
        if not line:
            continue
        X_str, C = line
        
        # Convert 1-based index from input to 0-based index for list access
        idx = int(X_str) - 1

        # If the character does not change, the count remains the same.
        if s_list[idx] == C:
            sys.stdout.write(str(count) + '
')
            continue

        # A change at `idx` can only affect "ABC" substrings starting at `idx-2`, `idx-1`, or `idx`.
        # Using a set conveniently handles potential out-of-bounds indices and duplicates.
        indices_to_check = {idx - 2, idx - 1, idx}

        # 1. Decrement count for any "ABC" instances that will be destroyed by the change.
        for i in indices_to_check:
            if is_abc(i):
                count -= 1

        # 2. Apply the character change to the list.
        s_list[idx] = C

        # 3. Increment count for any new "ABC" instances created by the change.
        for i in indices_to_check:
            if is_abc(i):
                count += 1
        
        # 4. Print the final count for this query.
        sys.stdout.write(str(count) + '
')

# Execute the solution
solve()