def main():
    import sys
    data = sys.stdin.read().strip().split()
    N, L, R = map(int, data)
    
    # Create the initial sequence
    A = list(range(1, N + 1))
    
    # Reverse the subarray from L to R (1-indexed)
    A[L-1:R] = reversed(A[L-1:R])
    
    # Print the result
    print(*A)

if __name__ == "__main__":
    main()