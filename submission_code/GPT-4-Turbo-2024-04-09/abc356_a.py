# YOUR CODE HERE
def main():
    import sys
    input = sys.stdin.read
    data = input().strip()
    N, L, R = map(int, data.split())
    
    # Create the initial sequence A
    A = list(range(1, N + 1))
    
    # Reverse the segment from L to R (inclusive)
    # Note: L and R are 1-based indices, Python uses 0-based indexing
    A[L-1:R] = reversed(A[L-1:R])
    
    # Print the modified sequence
    print(" ".join(map(str, A)))

if __name__ == "__main__":
    main()