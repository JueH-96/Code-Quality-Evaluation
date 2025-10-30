def reverse_subarray(N, L, R):
    # Create the initial sequence A
    A = list(range(1, N + 1))
    
    # Reverse the subarray from L to R (adjusting for 0-indexing)
    A[L-1:R] = reversed(A[L-1:R])
    
    # Return the modified sequence as a space-separated string
    return ' '.join(map(str, A))

# Read input from stdin
N, L, R = map(int, input().split())

# Print the result
print(reverse_subarray(N, L, R))