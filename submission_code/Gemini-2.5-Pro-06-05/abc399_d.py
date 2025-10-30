import sys

def run_solution():
    """
    Reads all test cases from stdin and runs the solver for each.
    """
    try:
        # Read the number of test cases
        t_str = sys.stdin.readline()
        if not t_str:
            return  # Handle empty input
        T = int(t_str)
    except (ValueError, IndexError):
        return

    for _ in range(T):
        solve_one_case()

def solve_one_case():
    """
    Solves a single test case.
    """
    try:
        # Read N and array A for the current test case
        N = int(sys.stdin.readline())
        A = list(map(int, sys.stdin.readline().split()))
    except (ValueError, IndexError):
        return # Malformed input for this case

    # --- Start of the core algorithm ---

    # 1. Pre-computation
    # pos[k] stores the two 0-based indices of value k
    pos = [[] for _ in range(N + 1)]
    for i, val in enumerate(A):
        pos[val].append(i)

    # is_adjacent[k] is True if couple k is initially adjacent
    is_adjacent = [False] * (N + 1)
    for k in range(1, N + 1):
        # Problem statement guarantees each number appears twice
        if pos[k][1] == pos[k][0] + 1:
            is_adjacent[k] = True

    # partner[i] stores the index of the other person of the same couple as A[i]
    partner = [0] * (2 * N)
    for k in range(1, N + 1):
        l, r = pos[k]
        partner[l] = r
        partner[r] = l

    # 2. Iterate through adjacent elements in A and count valid pairs
    found_pairs = set()
    for i in range(2 * N - 1):
        c1 = A[i]
        c2 = A[i+1]

        if c1 == c2:
            continue
        
        # Condition: neither couple was originally sitting next to each other
        if is_adjacent[c1] or is_adjacent[c2]:
            continue

        # Condition: can be made adjacent by swapping.
        # This holds if the partners of A[i] and A[i+1] are also adjacent.
        partner_of_i_idx = partner[i]
        partner_of_i_plus_1_idx = partner[i+1]

        if abs(partner_of_i_idx - partner_of_i_plus_1_idx) == 1:
            # We found a valid pair of couples (c1, c2).
            # Store in a set of sorted tuples to handle order and duplicates.
            pair = tuple(sorted((c1, c2)))
            found_pairs.add(pair)
            
    # 3. Print the result
    print(len(found_pairs))

# It's good practice to wrap the execution in a main check.
if __name__ == "__main__":
    run_solution()