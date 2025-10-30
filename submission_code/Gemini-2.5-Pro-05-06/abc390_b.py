import sys

def main():
    N = int(sys.stdin.readline())
    A = list(map(int, sys.stdin.readline().split()))

    is_geometric = True
    
    # The loop iterates from i = 0 up to N-3.
    # This means it requires at least 3 elements (A[0], A[1], A[2]) for the first check.
    # If N < 3 (i.e., N=2, given N >= 2 constraint), N-2 will be 0.
    # range(0) is an empty sequence, so the loop will not execute.
    # In this case (N=2), is_geometric remains True, which is correct.
    for i in range(N - 2):
        # We need to check if A[i+1]/A[i] == A[i+2]/A[i+1].
        # To avoid floating-point arithmetic and since A_j are positive integers,
        # we cross-multiply: A[i+1] * A[i+1] == A[i] * A[i+2].
        # If this condition is not met for any triplet, it's not a geometric progression.
        
        # Note: Python's integers support arbitrary precision, so intermediate
        # products up to 10^18 (from (10^9)^2) are handled correctly.
        if A[i] * A[i+2] != A[i+1] * A[i+1]:
            is_geometric = False
            break
            
    if is_geometric:
        sys.stdout.write("Yes
")
    else:
        sys.stdout.write("No
")

if __name__ == '__main__':
    main()