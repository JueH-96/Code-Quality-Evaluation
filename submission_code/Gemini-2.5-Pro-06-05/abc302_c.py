import sys
from itertools import permutations

def solve():
    """
    Reads input, solves the problem, and prints the output.
    """
    # Read problem parameters N (number of strings) and M (length of each string)
    # from the first line of standard input.
    try:
        N, M = map(int, sys.stdin.readline().split())
        # Read the N strings into a list.
        S = [sys.stdin.readline().strip() for _ in range(N)]
    except (IOError, ValueError):
        # This case is unlikely given the problem constraints (N>=2),
        # but handles potential empty or malformed input.
        print("No")
        return

    # The problem asks if there's an arrangement of strings where adjacent strings
    # differ by exactly one character. This is equivalent to finding a Hamiltonian path
    # in a graph where strings are vertices and an edge indicates a 1-character difference.

    # Since N is very small (2 <= N <= 8), we can check all N! permutations.
    # For N=8, 8! = 40320, which is computationally feasible.

    # Generate and iterate through all permutations of the input strings.
    for p in permutations(S):
        # 'p' is a tuple of strings, representing a candidate sequence.
        # Assume this path is valid until a counterexample is found.
        is_valid_path = True
        
        # Check each adjacent pair in the sequence.
        for i in range(N - 1):
            s1 = p[i]
            s2 = p[i+1]
            
            # Count the number of differing characters between s1 and s2.
            # A generator expression with sum() is a concise way to do this.
            diff_count = sum(1 for j in range(M) if s1[j] != s2[j])
            
            # If the difference is not exactly 1, this permutation is invalid.
            if diff_count != 1:
                is_valid_path = False
                break  # Stop checking this permutation and move to the next.
        
        # If the inner loop completed, we have found a valid sequence.
        if is_valid_path:
            print("Yes")
            # Terminate the program as a solution has been found.
            return

    # If we iterate through all permutations and none are valid,
    # no such arrangement is possible.
    print("No")

solve()