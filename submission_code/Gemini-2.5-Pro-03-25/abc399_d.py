# YOUR CODE HERE
import sys

# Read number of test cases
T = int(sys.stdin.readline())

# Process each test case
for _ in range(T):
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    # Store positions (0-based indices) for each number
    # Using list of lists since keys are integers 1..N. This is potentially faster than dict for large N.
    pos = [[] for _ in range(N + 1)] 
    for i in range(2 * N):
        val = A[i]
        # val guaranteed to be in [1, N] by constraints, so no bounds check needed.
        pos[val].append(i) 

    # Store Left (first occurrence index) and Right (second occurrence index) positions using arrays
    L = [0] * (N + 1)
    R = [0] * (N + 1)
    # Store whether a number's occurrences are adjacent using a boolean array
    is_adj = [False] * (N + 1)
    
    # Calculate L, R, and adjacency status for each number k from 1 to N
    for k in range(1, N + 1):
         # Each k from 1 to N appears exactly twice according to constraints.
         # pos[k] will contain the two 0-based indices.
         L[k] = pos[k][0]
         R[k] = pos[k][1]
         # Check if the indices are consecutive
         if R[k] == L[k] + 1:
             is_adj[k] = True
         # Default initialization is False, so no else needed.

    # Initialize count of valid pairs (a, b)
    total_count = 0

    # Iterate through all adjacent pairs of elements in the sequence A
    # i ranges from 0 to 2N-2
    for i in range(2 * N - 1):
        # Get the values at adjacent positions i and i+1
        a = A[i]
        b = A[i+1]

        # We are looking for pairs of *distinct* numbers (a, b)
        if a == b:
            continue

        # Consider the case where the pair (A[i], A[i+1]) forms the start (L[a], L[b])
        # This corresponds to a pattern starting with 'a b ...'
        if i == L[a] and (i + 1) == L[b]:
            
            # Check for Case A1: Structure like 'a b ... b a'
            # The condition is that the four positions are L[a], L[a]+1, R[b], R[b]+1.
            # This requires the endpoint condition R[a] == R[b] + 1.
            if R[a] == R[b] + 1:
                 # Conditions require both a and b to be non-adjacent.
                 # In Case A1 (L[b] = L[a]+1, R[a] = R[b]+1), 'a' spans a larger interval containing 'b'.
                 # 'a' is guaranteed non-adjacent (R[a] - L[a] = R[b]-L[b]+2 >= 3).
                 # We only need to check if 'b' is non-adjacent.
                 if not is_adj[b]: 
                     # Count pairs (a, b) such that a < b
                     if a < b: 
                         total_count += 1

            # Check for Case B1: Structure like 'a b ... a b'
            # The condition is that the four positions are L[a], L[a]+1, R[a], R[a]+1.
            # This requires the endpoint condition R[b] == R[a] + 1.
            if R[b] == R[a] + 1:
                 # Conditions require both a and b to be non-adjacent.
                 # In Case B1 (L[b] = L[a]+1, R[b] = R[a]+1), 'a' and 'b' interlace and span intervals of same length.
                 # 'a' is non-adjacent if and only if 'b' is non-adjacent.
                 # We check if 'a' is non-adjacent.
                 if not is_adj[a]: 
                     # Count pairs (a, b) such that a < b
                     if a < b:
                         total_count += 1

        # Consider the case where the pair (A[i], A[i+1]) forms the start (L[b], L[a])
        # This corresponds to a pattern starting with 'b a ...'
        if i == L[b] and (i + 1) == L[a]:
            
             # Check for Case A2: Structure like 'b a ... a b'
             # The condition is that the four positions are L[b], L[b]+1, R[a], R[a]+1.
             # This requires the endpoint condition R[b] == R[a] + 1.
             if R[b] == R[a] + 1:
                  # Conditions require both a and b to be non-adjacent.
                  # In Case A2 (L[a] = L[b]+1, R[b] = R[a]+1), 'b' spans a larger interval containing 'a'.
                  # 'b' is guaranteed non-adjacent. We only need to check if 'a' is non-adjacent.
                  if not is_adj[a]:
                      # Count pairs (a, b) such that a < b
                      if a < b: 
                          total_count += 1
             
             # Check for Case B2: Structure like 'b a ... b a'
             # The condition is that the four positions are L[b], L[b]+1, R[b], R[b]+1.
             # This requires the endpoint condition R[a] == R[b] + 1.
             if R[a] == R[b] + 1:
                   # Conditions require both a and b to be non-adjacent.
                   # In Case B2 (L[a] = L[b]+1, R[a] = R[b]+1), 'a' and 'b' interlace and span intervals of same length.
                   # 'a' is non-adjacent if and only if 'b' is non-adjacent.
                   # We check if 'b' is non-adjacent.
                   if not is_adj[b]:
                       # Count pairs (a, b) such that a < b
                       if a < b: 
                           total_count += 1

    # Print the final count for the current test case
    print(total_count)