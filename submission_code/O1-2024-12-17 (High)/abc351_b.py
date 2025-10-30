def main():
    import sys
    input_data = sys.stdin.read().strip().split()
    
    N = int(input_data[0])
    A = input_data[1:N+1]
    B = input_data[N+1:2*N+1]
    
    # Find the single position (i, j) where A and B differ.
    for i in range(N):
        for j in range(N):
            if A[i][j] != B[i][j]:
                print(i+1, j+1)
                return

# Do not forget to call main!
if __name__ == "__main__":
    main()