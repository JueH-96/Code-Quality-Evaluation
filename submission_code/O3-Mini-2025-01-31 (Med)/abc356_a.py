def main():
    import sys
    input = sys.stdin.readline
    # Read the input values.
    N, L, R = map(int, input().strip().split())
    
    # Create the sequence A = (1, 2, ..., N).
    A = list(range(1, N+1))
    
    # Reverse the segment from L to R, noting that list indices are 0-based.
    A[L-1:R] = A[L-1:R][::-1]
    
    # Print the sequence separated by spaces.
    print(" ".join(map(str, A)))

if __name__ == "__main__":
    main()