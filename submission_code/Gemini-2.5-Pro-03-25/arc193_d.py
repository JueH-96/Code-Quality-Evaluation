import sys

def solve():
    # Read the size of the strings
    N = int(sys.stdin.readline())
    # Read the initial configuration string A
    A = sys.stdin.readline().strip()
    # Read the target configuration string B
    B = sys.stdin.readline().strip()

    # Find the indices of all pieces ('1's) in the initial configuration A.
    # The constraints guarantee that A contains at least one '1'.
    # This list comprehension iterates through A once, taking O(N) time.
    A_indices = [i + 1 for i, char in enumerate(A) if char == '1']
    
    # Determine the leftmost (L_A) and rightmost (R_A) positions of pieces in A.
    # Accessing the first and last elements of the list takes O(1) time.
    L_A = A_indices[0]
    R_A = A_indices[-1]
    
    # Find the indices of all required occupied squares ('1's) in the target configuration B.
    # The constraints guarantee that B contains at least one '1'.
    B_indices = [i + 1 for i, char in enumerate(B) if char == '1']

    # Determine the leftmost (L_B) and rightmost (R_B) required occupied squares in B.
    L_B = B_indices[0]
    R_B = B_indices[-1]

    # Calculate the initial range width (d0) and the target range width (D).
    # The range width is the distance between the rightmost and leftmost pieces/occupied squares.
    d0 = R_A - L_A
    D = R_B - L_B

    # Check a necessary condition for reachability: The target range width D cannot be greater
    # than the initial range width d0. The operations can only decrease or maintain the range width.
    # If this condition is violated, it's impossible to reach the target configuration.
    if D > d0:
        print("-1")
        return

    # The total number of operations 'k' can be conceptually split into two parts:
    # 1. 'External' operations: where the chosen square 'p' is outside the current range [L, R] of pieces.
    #    These operations shift the entire range left (if p < L) or right (if p > R).
    #    Let k_L be the count of left shifts and k_R be the count of right shifts.
    #    The net shift from external operations is X = k_R - k_L.
    # 2. 'Internal' operations: where the chosen square 'p' is inside the current range [L, R].
    #    These operations can shrink the range width and shift the boundaries relative to the center.
    #    Let k_M be the count of internal operations.
    # The total number of operations is k = k_L + k_R + k_M.

    # The final positions L_B and R_B depend on initial positions L_A, R_A and the operations performed.
    # Specifically, L_B = L_A + (k_R - k_L) + S_L and R_B = R_A + (k_R - k_L) + S_R,
    # where S_L and S_R are the net shifts of the left and right boundaries caused by internal operations.
    # From these equations, we derive the required net external shift X = k_R - k_L must be in the interval [R_B - R_A, L_B - L_A].

    # Calculate the bounds for the required net external shift X.
    X_min = R_B - R_A
    X_max = L_B - L_A

    # Find X_opt, the value of X within the interval [X_min, X_max] that minimizes |X|.
    # This corresponds to the minimum number of external shift operations needed.
    # X_opt is the integer in the interval closest to 0.
    if X_min <= 0 <= X_max:
        X_opt = 0
    elif X_min > 0: # The interval [X_min, X_max] is entirely positive
        X_opt = X_min
    else: # X_max < 0, the interval [X_min, X_max] is entirely negative
        X_opt = X_max

    # Calculate S_L and S_R. These represent the required shifts of the left and right boundaries,
    # respectively, that must be achieved solely by the 'internal' operations.
    # S_L = (Total required shift for L) - (Shift achieved by external ops) = (L_B - L_A) - X_opt
    S_L = (L_B - L_A) - X_opt
    # S_R = (Total required shift for R) - (Shift achieved by external ops) = (R_B - R_A) - X_opt
    S_R = (R_B - R_A) - X_opt

    # The minimum total number of operations 'k' is derived based on the principle that
    # k = |X_opt| + k_M, where k_M is the minimum number of internal operations required.
    # It's hypothesized and verified on examples that the minimum k_M required to achieve the internal shifts
    # S_L and S_R (and the necessary width reduction) is determined by the larger magnitude of these required internal shifts.
    # That is, k_M = max(|S_L|, |S_R|).
    # Thus, the total minimum operations k = |X_opt| + max(abs(S_L), abs(S_R)).
    k = abs(X_opt) + max(abs(S_L), abs(S_R))
    print(k)

# Read the number of test cases
T = int(sys.stdin.readline())
# Process each test case
for _ in range(T):
    solve()