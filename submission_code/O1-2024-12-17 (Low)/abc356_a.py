def main():
    # Read integers N, L, R from stdin
    N, L, R = map(int, input().split())
    
    # Initialize the sequence A
    A = list(range(1, N+1))
    
    # Reverse the portion from L-th to R-th elements (1-indexed)
    A[L-1:R] = reversed(A[L-1:R])
    
    # Print the resulting sequence
    print(" ".join(map(str, A)))

# Do not forget to call the main function
main()