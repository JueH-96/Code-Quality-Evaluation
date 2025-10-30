import sys

def solve():
    # Read N (length of string) and Q (number of queries)
    N, Q = map(int, sys.stdin.readline().split())

    # Read the initial string S and convert it to a list of characters.
    # Using a list allows O(1) modification of individual characters,
    # which is crucial for performance. .strip() removes trailing newline.
    S_list = list(sys.stdin.readline().strip())

    # Initialize the current count of "ABC" occurrences.
    current_abc_count = 0

    # Helper function to check if the substring starting at `idx` is "ABC".
    # It correctly handles boundary conditions.
    def check_abc(idx):
        # A substring "ABC" requires 3 characters.
        # If it starts at `idx`, it uses S_list[idx], S_list[idx+1], S_list[idx+2].
        # For these indices to be valid, `idx` must be non-negative
        # and `idx+2` must be within the string bounds (less than N).
        # This combined condition is 0 <= idx <= N - 3.
        if 0 <= idx <= N - 3:
            return S_list[idx] == 'A' and S_list[idx+1] == 'B' and S_list[idx+2] == 'C'
        return False # Out of bounds, so cannot be "ABC"

    # Calculate the initial count of "ABC" in the string.
    # We iterate through all possible starting positions for "ABC".
    for i in range(N - 2): # Loop from 0 up to N-3 (inclusive)
        if check_abc(i):
            current_abc_count += 1

    # List to store the results of each query. This allows printing all
    # results at once efficiently using sys.stdout.write.
    results = []

    # Process each of the Q queries
    for _ in range(Q):
        # Read the query: character position X (1-indexed) and new character C
        X, C = sys.stdin.readline().split()
        X = int(X)
        idx = X - 1 # Convert 1-based index X to 0-based index `idx`

        # When S_list[idx] is about to change, it can affect at most three "ABC" substrings:
        # 1. An "ABC" substring that ends at `idx` (i.e., starts at `idx-2`). Pattern: A B C
        #                                                                        idx-2 idx-1 idx
        # 2. An "ABC" substring where `S_list[idx]` is the 'B' character (i.e., starts at `idx-1`). Pattern:   A B C
        #                                                                                                    idx-1 idx idx+1
        # 3. An "ABC" substring that starts at `idx`. Pattern:     A B C
        #                                                        idx idx+1 idx+2

        # Step 1: Before changing S_list[idx], check if any of these three patterns
        # currently exist. If they do, decrement the count, as they might be broken.
        if check_abc(idx - 2): # Check for ABC ending at idx
            current_abc_count -= 1
        if check_abc(idx - 1): # Check for ABC with B at idx
            current_abc_count -= 1
        if check_abc(idx):     # Check for ABC starting at idx
            current_abc_count -= 1

        # Step 2: Perform the character change
        S_list[idx] = C

        # Step 3: After changing S_list[idx], check if any of the same three patterns
        # now exist (or are newly formed). If they do, increment the count.
        if check_abc(idx - 2):
            current_abc_count += 1
        if check_abc(idx - 1):
            current_abc_count += 1
        if check_abc(idx):
            current_abc_count += 1
        
        # Store the current count (as a string) for output.
        results.append(str(current_abc_count))
    
    # Print all accumulated results, each on a new line.
    sys.stdout.write("
".join(results) + "
")

# Call the solve function to execute the program logic
solve()