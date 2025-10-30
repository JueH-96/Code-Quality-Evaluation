# YOUR CODE HERE
import sys

def solve():
    """
    Solves a single test case.
    """
    try:
        N_str = sys.stdin.readline()
        if not N_str: return
        N = int(N_str)
        A = list(map(int, sys.stdin.readline().split()))
    except (IOError, ValueError):
        # Handle potential empty lines or malformed input at the end of file
        return

    # 1. Calculate the invariant sum S.
    total_sum = sum(A)

    # 2. Determine the elements of the "flattest" non-decreasing sequence B*.
    # B* consists of N-r elements of value q and r elements of value q+1.
    q = total_sum // N
    r = total_sum % N

    # 3. Check if prefix sums of A are <= prefix sums of B*.
    prefix_sum_A = 0
    prefix_sum_B_star = 0
    
    possible = True
    for i in range(N):
        prefix_sum_A += A[i]
        
        # Calculate B*[i] and add it to its prefix sum.
        # The first N-r elements (0-indexed) are q, the rest are q+1.
        if i < N - r:
            prefix_sum_B_star += q
        else:
            prefix_sum_B_star += q + 1
        
        # The condition is that for any prefix, the sum of A's elements
        # cannot exceed the sum of B*'s elements.
        if prefix_sum_A > prefix_sum_B_star:
            possible = False
            break
    
    # At i = N-1, the prefix sums must be equal (total_sum), so
    # prefix_sum_A > prefix_sum_B_star will not be true if it hasn't been
    # for i < N-1. The `break` is safe.
            
    if possible:
        sys.stdout.write("Yes
")
    else:
        sys.stdout.write("No
")

def main():
    """
    Main function to handle multiple test cases.
    """
    try:
        T_str = sys.stdin.readline()
        if not T_str: return
        T = int(T_str)
        for _ in range(T):
            solve()
    except (IOError, ValueError):
        return

if __name__ == "__main__":
    main()