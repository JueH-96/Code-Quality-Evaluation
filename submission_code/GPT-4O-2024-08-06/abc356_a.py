# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    
    # Read the input values
    N, L, R = map(int, data.split())
    
    # Create the initial sequence A
    A = list(range(1, N + 1))
    
    # Perform the reverse operation on the subarray from L to R
    # Note: L and R are 1-based indices, so we need to adjust them to 0-based
    A[L-1:R] = reversed(A[L-1:R])
    
    # Print the resulting sequence
    print(' '.join(map(str, A)))

if __name__ == "__main__":
    main()