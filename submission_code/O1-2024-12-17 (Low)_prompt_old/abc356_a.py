def solve():
    import sys
    
    data = sys.stdin.read().strip().split()
    N, L, R = map(int, data)

    # Create the sequence A
    A = list(range(1, N+1))
    
    # Reverse the L-th through R-th elements
    # Note that A is 0-indexed, so we need to adjust indices by -1
    A[L-1:R] = reversed(A[L-1:R])
    
    # Print the resulting sequence
    print(" ".join(map(str, A)))

def main():
    solve()

if __name__ == "__main__":
    main()