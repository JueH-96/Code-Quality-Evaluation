# YOUR CODE HERE
def main():
    import sys
    
    # Read input values
    N, L, R = map(int, sys.stdin.readline().split())
    
    # Create the sequence A = (1, 2, ..., N)
    A = list(range(1, N + 1))
    
    # Reverse the sublist from L-th to R-th elements (1-based indexing)
    A[L-1:R] = reversed(A[L-1:R])
    
    # Print the resulting sequence
    print(" ".join(map(str, A)))

# Call the main function
main()