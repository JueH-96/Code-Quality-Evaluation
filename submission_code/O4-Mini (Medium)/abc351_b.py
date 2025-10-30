def main():
    import sys
    data = sys.stdin.read().split()
    # First token is N
    N = int(data[0])
    # Next N tokens are rows of grid A, then N tokens for grid B
    A = data[1:1+N]
    B = data[1+N:1+2*N]
    
    for i in range(N):
        row_a = A[i]
        row_b = B[i]
        for j in range(N):
            if row_a[j] != row_b[j]:
                # Output 1-based indices
                print(i+1, j+1)
                return

if __name__ == "__main__":
    main()