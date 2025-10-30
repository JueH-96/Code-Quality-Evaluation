def main():
    import sys
    
    # Read number of elements
    N_line = sys.stdin.readline()
    while N_line.strip() == '':
        N_line = sys.stdin.readline()
    N = int(N_line.strip())

    # Read the list of integers
    A_line = sys.stdin.readline()
    while A_line.strip() == '':
        A_line = sys.stdin.readline()
    A = list(map(int, A_line.strip().split()))
    
    # Ensure we have exactly N integers; in rare cases the list might span several lines
    while len(A) < N:
        A.extend(map(int, sys.stdin.readline().strip().split()))
    
    # Compute products B_i = A_i * A_{i+1}
    B = [A[i] * A[i + 1] for i in range(N - 1)]
    
    # Output
    print(' '.join(map(str, B)))

if __name__ == "__main__":
    main()