import sys

def solve():
    # Read N, L, R. N is not explicitly used in this Pythonic solution
    # as list length is obtained when reading A_values, but it's part of input.
    _N, L, R = map(int, sys.stdin.readline().split())
    
    # Read the sequence A.
    A_values = list(map(int, sys.stdin.readline().split()))
    
    # For each element a_val in A_values, compute the corresponding x_val.
    # x_val is the integer in the range [L, R] that is closest to a_val.
    # This can be achieved by clamping a_val to the range [L, R].
    # The clamping operation can be expressed as max(L, min(a_val, R)).
    
    # Example walkthrough of max(L, min(a_val, R)):
    # Suppose L=5, R=10.
    # If a_val = 3 (less than L):
    #   min(3, 10) is 3.
    #   max(5, 3) is 5. So, x_val = 5 (which is L). This is correct.
    # If a_val = 7 (within [L, R]):
    #   min(7, 10) is 7.
    #   max(5, 7) is 7. So, x_val = 7 (which is a_val). This is correct.
    # If a_val = 12 (greater than R):
    #   min(12, 10) is 10.
    #   max(5, 10) is 10. So, x_val = 10 (which is R). This is correct.
    
    # Use a list comprehension to generate all X values.
    X_results = [max(L, min(a_val, R)) for a_val in A_values]
    
    # Print the results, space-separated, followed by a newline.
    # map(str, X_results) converts all integers in X_results to strings.
    # " ".join(...) joins these strings with a space in between.
    sys.stdout.write(" ".join(map(str, X_results)) + "
")

# Standard boilerplate to call the main function.
if __name__ == '__main__':
    solve()